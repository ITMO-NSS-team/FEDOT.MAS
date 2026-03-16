from __future__ import annotations

import time
from typing import Optional

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.events import Event
from google.adk.plugins import BasePlugin
from google.adk.runners import InvocationContext
from google.genai import types

from fedotmas.common.logging import get_logger

_log = get_logger("fedotmas.plugins.logging")

_WORKFLOW_PREFIXES = ("seq_", "par_", "loop_")


def _is_workflow_node(name: str) -> bool:
    return name.startswith(_WORKFLOW_PREFIXES)


class LoggingPlugin(BasePlugin):
    """Default FEDOT.MAS plugin that logs agent lifecycle and event details.

    Consolidates logging previously scattered across ``_ppline_utils``,
    ``builder``, and ``runner``.  Always returns ``None`` so it never
    short-circuits other plugins.
    """

    def __init__(self) -> None:
        super().__init__(name="fedotmas_logging")
        self._agent_start: dict[str, float] = {}

    async def before_agent_callback(
        self, *, agent: BaseAgent, callback_context: CallbackContext
    ) -> Optional[types.Content]:
        if not _is_workflow_node(agent.name):
            _log.info("Agent started | name={}", agent.name)
        self._agent_start[agent.name] = time.monotonic()
        return None

    async def after_agent_callback(
        self, *, agent: BaseAgent, callback_context: CallbackContext
    ) -> Optional[types.Content]:
        t0 = self._agent_start.pop(agent.name, None)
        if t0 is not None and not _is_workflow_node(agent.name):
            _log.info(
                "Agent done | name={} elapsed={:.1f}s",
                agent.name,
                time.monotonic() - t0,
            )
        return None

    async def on_event_callback(
        self, *, invocation_context: InvocationContext, event: Event
    ) -> Optional[Event]:
        if event.partial:
            return None

        # Tool calls
        for fc in event.get_function_calls():
            _log.info(
                "Tool call | agent={} tool={} args={}",
                event.author,
                fc.name,
                fc.args,
            )

        # Tool responses
        for fr in event.get_function_responses():
            resp_str = str(fr.response)[:200] if fr.response else ""
            if fr.response and "error" in resp_str.lower():
                _log.warning(
                    "Tool error | agent={} tool={} response={}",
                    event.author,
                    fr.name,
                    resp_str,
                )
            else:
                _log.info(
                    "Tool result | agent={} tool={}",
                    event.author,
                    fr.name,
                )

        # Token usage
        if event.usage_metadata:
            um = event.usage_metadata
            prompt = um.prompt_token_count or 0
            completion = um.candidates_token_count or 0
            if prompt or completion:
                _log.info(
                    "Tokens | agent={} prompt={} completion={}",
                    event.author,
                    prompt,
                    completion,
                )

        # Text response (no function calls)
        if event.content and event.content.parts and not event.get_function_calls():
            texts = [p.text for p in event.content.parts if p.text]
            if texts:
                preview = texts[0][:200]
                _log.trace("Response | agent={} text={}", event.author, preview)

        # State changes
        if event.actions.state_delta:
            for key, value in event.actions.state_delta.items():
                if value is None or (isinstance(value, str) and not value.strip()):
                    _log.warning(
                        "Empty output | agent={} key='{}'",
                        event.author,
                        key,
                    )
            _log.info(
                "State update | agent={} keys={}",
                event.author,
                list(event.actions.state_delta.keys()),
            )

        return None
