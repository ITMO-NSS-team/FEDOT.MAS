"""Поток событий выполнения: плагин ADK и обёртка SSE.

Цикл выдачи событий был трижды скопирован в обработчиках generate_stream, run и
judge_stream — вместе с пульсом и отменой задачи. Теперь он один.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Awaitable, Callable, Optional

from fastapi.responses import StreamingResponse
from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.events import Event
from google.adk.plugins import BasePlugin
from google.adk.runners import InvocationContext
from google.genai import types

from .agent_names import AgentNames
from .config import SSE_HEARTBEAT, WORKFLOW_PREFIXES


def rubber_validation_result(value, depth=0):
    """Extract structured predictor diagnostics through MCP response wrappers."""
    if depth > 6:
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return None
    if isinstance(value, dict):
        if value.get("status") == "prediction_completed" and isinstance(value.get("recipe_validation"), dict):
            return value["recipe_validation"]
        children = value.values()
    elif isinstance(value, list):
        children = value
    else:
        return None
    for child in children:
        found = rubber_validation_result(child, depth + 1)
        if found is not None:
            return found
    return None


class StreamPlugin(BasePlugin):
    """Пробрасывает события выполнения в очередь — интерфейс читает её как SSE.

    *names* переводит имена запуска в имена сценария: MAS исполняется под латинскими
    именами (см. agent_names), а интерфейс рисует граф и ленту по русским.
    """

    def __init__(self, queue: asyncio.Queue, names: AgentNames | None = None) -> None:
        super().__init__(name="gui_stream")
        self.queue = queue
        self.names = names or AgentNames()
        self._start: dict[str, float] = {}
        self.tokens = 0

    def _put(self, payload: dict[str, Any]) -> None:
        self.queue.put_nowait(payload)

    def _is_workflow(self, name: str) -> bool:
        # Латинское имя агента MAS может случайно начаться с «par_» — это всё равно агент
        return not self.names.is_agent(name) and name.startswith(WORKFLOW_PREFIXES)

    async def before_agent_callback(
        self, *, agent: BaseAgent, callback_context: CallbackContext
    ) -> Optional[types.Content]:
        if self._is_workflow(agent.name):
            return None
        self._start[agent.name] = time.monotonic()
        # Вход агента: инструкция и то, что уже лежит в состоянии от предшественников.
        # Без этого «ход выполнения» показывает только реплики, и непонятно,
        # на основании чего агент их выдал.
        instruction = str(getattr(agent, "instruction", "") or "")
        try:
            state = dict(callback_context.state.to_dict())
        except Exception:
            state = {}
        incoming = {k: self.names.text(str(v)[:4000]) for k, v in state.items() if k != "user_query"}
        self._put({
            "type": "agent_start",
            "agent": self.names.name(agent.name),
            "instruction": self.names.text(instruction[:8000]),
            "incoming": incoming,
        })
        return None

    async def after_agent_callback(
        self, *, agent: BaseAgent, callback_context: CallbackContext
    ) -> Optional[types.Content]:
        t0 = self._start.pop(agent.name, None)
        if t0 is None or self._is_workflow(agent.name):
            return None
        output_key = getattr(agent, "output_key", None)
        produced = ""
        if output_key:
            try:
                produced = str(callback_context.state.to_dict().get(output_key, ""))
            except Exception:
                produced = ""
        self._put({"type": "agent_done", "agent": self.names.name(agent.name),
                   "ms": int((time.monotonic() - t0) * 1000),
                   "output_key": output_key, "output": self.names.text(produced[:12000])})
        return None

    async def on_event_callback(
        self, *, invocation_context: InvocationContext, event: Event
    ) -> Optional[Event]:
        if getattr(event, "partial", False):
            return None
        author = self.names.name(getattr(event, "author", "?"))

        usage = getattr(event, "usage_metadata", None)
        tokens = 0
        if usage is not None:
            tokens = (getattr(usage, "prompt_token_count", 0) or 0) + (
                getattr(usage, "candidates_token_count", 0) or 0
            )
            self.tokens += tokens

        for fc in event.get_function_calls():
            args = fc.args or {}
            if self.names.is_agent(fc.name):
                # Исполнитель MAS вызывается у координатора как инструмент с именем агента.
                # Для интерфейса это передача задачи — показываем её как transfer_to_agent.
                self._put({"type": "tool", "agent": author, "tool": "transfer_to_agent",
                           "target": self.names.name(fc.name),
                           "args": self.names.text(str(args)[:120])})
                continue
            target = args.get("agent_name") or args.get("agent") or ""
            self._put({"type": "tool", "agent": author, "tool": fc.name,
                       "target": self.names.name(str(target)),
                       "args": self.names.text(str(args)[:4000])})
        for fr in event.get_function_responses():
            raw_response = str(fr.response) if fr.response is not None else ""
            resp = raw_response[:12000]
            is_error = isinstance(fr.response, dict) and fr.response.get("isError") is True
            self._put({"type": "tool_result", "agent": author, "tool": self.names.name(fr.name),
                       "error": bool(is_error), "text": self.names.text(resp),
                       "rubber_validation": (rubber_validation_result(fr.response)
                                             if not is_error and "predict_rubber_properties" in fr.name else None),
                       "truncated": len(raw_response) > len(resp)})

        text = ""
        content = getattr(event, "content", None)
        if content and getattr(content, "parts", None):
            text = "".join(p.text or "" for p in content.parts if getattr(p, "text", None))
        if text.strip():
            self._put({"type": "text", "agent": author,
                       "text": self.names.text(text.strip()[:4000]), "tokens": tokens})
        elif tokens:
            self._put({"type": "tokens", "agent": author, "tokens": tokens})
        return None


def sse_stream(queue: asyncio.Queue, execute: Callable[[], Awaitable[None]]) -> StreamingResponse:
    """Отдаёт содержимое очереди браузеру как поток SSE.

    *execute* обязана закончить работу, положив в очередь None — это признак конца
    потока. Всё остальное уходит клиенту строками ``data:``.
    """

    async def events():
        task = asyncio.create_task(execute())
        try:
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=SSE_HEARTBEAT)
                except asyncio.TimeoutError:
                    # Пульс. Туннели и обратные прокси рвут молчащее соединение, а у
                    # судьи между событиями бывает минута тишины — на демонстрации это
                    # выглядело как «network error». Строка-комментарий по спецификации
                    # SSE клиентом игнорируется.
                    yield ": ping\n\n"
                    continue
                if item is None:
                    break
                yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
        finally:
            if not task.done():
                task.cancel()

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})
