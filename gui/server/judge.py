"""Судья: независимое сравнение ответа системы и ответа одной модели.

Судья работает как отдельная одноагентная система с песочницей — считать он обязан
вызовом инструмента, а не в уме. Ошибки в устном счёте уже приводили к неверным
вердиктам, поэтому здесь же живёт строгий повтор и ограничение на его время.
"""

from __future__ import annotations

import asyncio
import re

from fedotmas import MAW, MAWConfig
from fedotmas.common.logging import get_logger
from fedotmas.plugins import LoggingPlugin, UnknownToolRecoveryPlugin

from .config import (JUDGE_FALLBACK, JUDGE_MAX_TOKENS, JUDGE_MODEL,
                     JUDGE_RETRY_TIMEOUT, SAFE_TOOLS)
from .llm import client as _client
from .prompts import JUDGE_CONTENT, JUDGE_PROMPT
from .schemas import JudgeIn
from .streaming import StreamPlugin

_log = get_logger("gui.judge")


async def _ask_judge_direct(model: str, prompt: str, content: str) -> str:
    """Прямой вызов без ADK-агента — последний рубеж, когда агентный путь падает
    (наблюдалось: три APITimeoutError подряд на ровном месте). Без песочницы судья
    слабее в арифметике, но вердикт с оговоркой лучше отсутствия вердикта."""
    client, resolved = _client(model)
    resp = await client.chat.completions.create(
        model=resolved, max_tokens=JUDGE_MAX_TOKENS,
        messages=[{"role": "user", "content": f"{prompt}\n\n{content}"}])
    return (resp.choices[0].message.content or "").strip()


async def _ask_judge(model: str, prompt: str, content: str = "",
                     events: asyncio.Queue | None = None,
                     usage: dict | None = None) -> str:
    """Один прогон судьи. Судья работает агентом с песочницей: аудит показал, что в уме
    он ошибается на суммах длинных рядов и записывает верный чужой расчёт в ошибки."""
    if "sandbox-light" in SAFE_TOOLS:
        cfg = MAWConfig(**{
            "agents": [{
                "name": "судья",
                "description": "Сравнивает два ответа, пересчитывая числа в песочнице",
                # Правила — в инструкцию, сами ответы — в запрос: ADK подставляет в инструкцию
                # {переменные} из состояния, и фигурная скобка в чужом тексте роняет прогон.
                "instruction": prompt,
                "tools": ["sandbox-light"],
                "output_key": "вердикт",
                "model": model,
                "max_output_tokens": JUDGE_MAX_TOKENS,
            }],
            "pipeline": {"type": "agent", "agent_name": "судья"},
        })
        async def run_once(extra: str = "") -> tuple[str, int]:
            queue: asyncio.Queue = asyncio.Queue()
            stream = StreamPlugin(queue)
            system = MAW(worker_models=[model], mcp_servers=["sandbox-light"],
                         plugins=[LoggingPlugin(), UnknownToolRecoveryPlugin(), stream])
            calls = 0

            async def pump() -> None:
                """Проброс событий судьи наружу: без них интерфейс выглядит зависшим."""
                nonlocal calls
                while True:
                    item = await queue.get()
                    if item is None:
                        return
                    if isinstance(item, dict):
                        if item.get("type") == "tool":
                            calls += 1
                        if events is not None:
                            events.put_nowait(item)

            pumping = asyncio.create_task(pump())
            try:
                result = await system.build_and_run(
                    cfg, (content or "Вынеси вердикт по правилам из инструкции.") + extra)
            finally:
                queue.put_nowait(None)
                await pumping
            if usage is not None:
                usage["tokens"] = usage.get("tokens", 0) + stream.tokens
            state = result if isinstance(result, dict) else getattr(result, "state", {})
            return str(state.get("вердикт", "")).strip(), calls

        text, sandbox_calls = await run_once()
        # Повтор нужен не всегда: он оправдан, только когда судья ОБВИНЯЕТ сторону в ошибке
        # в числах, не пересчитав их. Если спора о числах нет, второй проход — чистая трата
        # времени (замер: медиана прогона 22 с, а повтор запускался в 70 случаях из 82).
        disputes = re.search(r"ошибк|неверн|не сходится|противореч|расхожден|занижен|завышен",
                             text or "", re.IGNORECASE)
        if text and sandbox_calls == 0 and disputes and any(ch.isdigit() for ch in text):
            # Аудит показал: без песочницы судья ошибается в сумме ряда и наказывает
            # правую сторону за несуществующее противоречие. Один строгий повтор.
            _log.warning("Судья не вызвал песочницу ни разу — строгий повтор | модель={}", model)
            # Повтор — это подстраховка, а не обязательный этап, и ждать его бесконечно
            # нельзя: замер показал проход на 170 с, который закончился вообще без вердикта
            # (в состоянии не появился ключ «вердикт»). Первый вердикт при этом уже есть.
            try:
                text2, calls2 = await asyncio.wait_for(
                    run_once("\n\nНАПОМИНАНИЕ: в прошлый раз ты не вызвал песочницу ни разу и "
                             "считал в уме. Так вердикт не принимается. Выполни каждый пересчёт "
                             "вызовом sandbox-light."),
                    timeout=JUDGE_RETRY_TIMEOUT)
            except asyncio.TimeoutError:
                _log.warning("Строгий повтор судьи не уложился в {} с — берём первый вердикт",
                             JUDGE_RETRY_TIMEOUT)
                text2, calls2 = "", 0
            if text2 and calls2 > 0:
                text = text2
            else:
                _log.info("Строгий повтор ничего не дал | вердикт={} вызовов={}",
                          bool(text2), calls2)
    else:
        client, resolved = _client(model)
        resp = await client.chat.completions.create(
            model=resolved, max_tokens=JUDGE_MAX_TOKENS,
            messages=[{"role": "user", "content": f"{prompt}\n\n{content}"}])
        text = (resp.choices[0].message.content or "").strip()
    if not text:
        # Рассуждающие модели иногда тратят весь бюджет вывода на размышления и отдают пустоту.
        _log.warning("Судья вернул пустой ответ | модель={}", model)
    return text


def _parse_winner(text: str) -> str | None:
    """«ПОБЕДИТЕЛЬ: А» — кириллицей или латиницей, возможно в звёздочках."""
    line = next((ln for ln in text.splitlines() if ln.strip().upper().startswith("ПОБЕДИТЕЛЬ")), "")
    head = line.upper()
    if re.search(r":\s*\**\s*(ОТВЕТ\s*)?[AА]\b", head):
        return "system"
    if re.search(r":\s*\**\s*(ОТВЕТ\s*)?[БB]\b", head):
        return "single"
    if "НИЧЬ" in head:
        return "tie"
    return None


async def _judge_impl(body: JudgeIn, events: asyncio.Queue | None = None) -> dict:
    """Сравнение двух ответов. События судьи уходят в очередь, если она передана."""
    prompt = JUDGE_PROMPT
    content = JUDGE_CONTENT.format(query=body.query, a=body.system_answer, b=body.single_answer)
    # Пустой ответ судьи раньше молча превращался в «ничью» — то есть в вердикт,
    # которого судья не выносил. Пробуем повтор, затем запасную модель, и только
    # потом честно сообщаем об ошибке.
    attempts = [body.model or JUDGE_MODEL, body.model or JUDGE_MODEL, JUDGE_FALLBACK]
    last_error: Exception | None = None
    usage: dict = {"tokens": 0}
    for attempt, model in enumerate(attempts, 1):
        try:
            text = await _ask_judge(model, prompt, content, events, usage)
        except Exception as exc:
            last_error = exc
            _log.warning("Судья не ответил | попытка={} модель={} | {}", attempt, model, exc)
            if attempt == len(attempts):        # агентный путь исчерпан — пробуем без ADK
                try:
                    text = await _ask_judge_direct(model, prompt, content)
                    _log.info("Судья ответил прямым вызовом без песочницы | модель={}", model)
                except Exception as exc2:
                    last_error = exc2
                    continue
            else:
                continue
        winner = _parse_winner(text) if text else None
        if winner is not None:
            return {"ok": True, "verdict": text, "winner": winner, "model": model,
                    "tokens": usage["tokens"]}
        if text:
            _log.warning("В ответе судьи нет строки победителя | модель={}", model)

    # Агентный путь исчерпан без вердикта (пустые ответы или текст без строки победителя) —
    # последний рубеж: прямой вызов без ADK. Раньше он срабатывал только при исключении,
    # и три пустых ответа подряд оставляли демонстрацию вовсе без вердикта.
    try:
        text = await _ask_judge_direct(JUDGE_FALLBACK, prompt, content)
        winner = _parse_winner(text) if text else None
        if winner is not None:
            _log.info("Вердикт вынесен прямым вызовом без песочницы | модель={}", JUDGE_FALLBACK)
            return {"ok": True, "verdict": text, "winner": winner,
                    "model": JUDGE_FALLBACK, "tokens": usage["tokens"]}
    except Exception as exc:
        last_error = exc

    if last_error is not None:
        return {"ok": False, "error": f"{type(last_error).__name__}: {last_error}"}
    return {"ok": False, "error": "судья не вернул вердикт: ответ пуст или без строки «ПОБЕДИТЕЛЬ»"}
