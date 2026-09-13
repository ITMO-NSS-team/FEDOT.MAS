"""Запись реального прогона FEDOT.MAS в сценарий для GUI.

Генерирует конфигурацию мета-агентом, исполняет её и печатает готовый блок
для `presets.js` — с настоящими текстами агентов, вызовами инструментов,
расходом токенов и временем.

Запуск (из корня репозитория FEDOT.MAS, где лежит .env с ключом):

    .venv/bin/python gui/tools/record_run.py \
        --task "Оцени риск отказа ..." \
        --query "Данные: ..." \
        --model openai/gpt-4.1-mini \
        --tools sandbox-light sequential-thinking \
        --id my_run --title "Реальный прогон" --domain "Нефтегаз · live" \
        --out preset.js

Готовый блок вставляется в массив `window.PRESETS` в `presets.js`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.events import Event
from google.adk.plugins import BasePlugin
from google.adk.runners import InvocationContext
from google.genai import types

os.environ.setdefault("FEDOTMAS_META_AGENT_MAX_OUTPUT_TOKENS", "8000")

from fedotmas import MAS, MAW
from fedotmas.plugins import LoggingPlugin

# Перевод имён агентов MAS общий с сервером стенда: gui/ нужен в пути импорта
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from server.agent_names import AgentNames, latinize_mas

WORKFLOW_PREFIXES = ("seq_", "par_", "loop_")


class TraceCollector(BasePlugin):
    """Собирает журнал запуска: тексты агентов, вызовы инструментов, токены, тайминги."""

    def __init__(self) -> None:
        super().__init__(name="trace_collector")
        self.steps: list[dict] = []
        self._start: dict[str, float] = {}
        self._tokens: dict[str, int] = {}
        self._tools: dict[str, list[str]] = {}
        self._order: list[str] = []
        self.t0 = time.monotonic()

    def take(self) -> list[dict]:
        """Забирает накопленные шаги и очищает коллектор (генерация и запуск считаются раздельно)."""
        steps, self.steps = self.steps, []
        self._start.clear()
        self._tokens.clear()
        self._tools.clear()
        self._order.clear()
        self.t0 = time.monotonic()
        return steps

    def _touch(self, name: str) -> None:
        if name not in self._order and not name.startswith(WORKFLOW_PREFIXES):
            self._order.append(name)

    async def before_agent_callback(
        self, *, agent: BaseAgent, callback_context: CallbackContext
    ) -> Optional[types.Content]:
        self._start[agent.name] = time.monotonic()
        self._touch(agent.name)
        return None

    async def after_agent_callback(
        self, *, agent: BaseAgent, callback_context: CallbackContext
    ) -> Optional[types.Content]:
        t0 = self._start.pop(agent.name, None)
        if t0 is None or agent.name.startswith(WORKFLOW_PREFIXES):
            return None
        self.steps.append(
            {
                "agent": agent.name,
                "ms": int((time.monotonic() - t0) * 1000),
                "started_at": round(t0 - self.t0, 2),
                "tokens": self._tokens.pop(agent.name, 0),
                "tools": self._tools.pop(agent.name, []),
            }
        )
        return None

    async def on_event_callback(
        self, *, invocation_context: InvocationContext, event: Event
    ) -> Optional[Event]:
        if getattr(event, "partial", False):
            return None
        author = getattr(event, "author", "?")
        usage = getattr(event, "usage_metadata", None)
        if usage is not None:
            self._tokens[author] = self._tokens.get(author, 0) + (
                (getattr(usage, "prompt_token_count", 0) or 0)
                + (getattr(usage, "candidates_token_count", 0) or 0)
            )
        for fc in event.get_function_calls():
            self._tools.setdefault(author, []).append(fc.name)
        return None


def _winner(verdict: str) -> str:
    """Разбор вердикта судьи. Модель пишет «ПОБЕДИТЕЛЬ: А» кириллицей — учитываем оба алфавита."""
    head = (verdict.strip().splitlines() or [""])[0].upper()
    if re.search(r":\s*[AА]\b", head):
        return "system"
    if re.search(r":\s*[БB]\b", head):
        return "single"
    return "tie"


JUDGE_PROMPT = """Ты — независимый судья. Сравни два ответа на один и тот же запрос.

ЗАПРОС:
{query}

ОТВЕТ А (мультиагентная система FEDOT.MAS):
{a}

ОТВЕТ Б (одна модель без системы):
{b}

Оцени по критериям: полнота, обоснованность, учёт ограничений из запроса, практическая применимость.
Ответь строго в формате:
ПОБЕДИТЕЛЬ: A | Б | ничья
ПОЧЕМУ: 2–4 предложения по-русски, с конкретными отличиями.
ЧЕГО НЕ ХВАТАЕТ В ПРОИГРАВШЕМ: одно предложение."""


def valid_name(name: str) -> str:
    """Имя агента должно быть корректным идентификатором Python — иначе ADK не соберёт систему."""
    cleaned = re.sub(r"\W", "_", (name or "").strip(), flags=re.UNICODE).strip("_") or "agent"
    if cleaned[0].isdigit():
        cleaned = "a_" + cleaned
    return cleaned if cleaned.isidentifier() else "agent"


def sanitize(config, kind: str):
    agents = list(getattr(config, "agents", None) or ([config.coordinator] + list(config.workers)))
    renames, keys = {}, {}
    for a in agents:
        new = valid_name(a.name)
        if new != a.name:
            renames[a.name] = new
            a.name = new
        key = getattr(a, "output_key", None)
        if key and valid_name(key) != key:
            keys[key] = valid_name(key)
            a.output_key = keys[key]
    for a in agents:
        for old, new in keys.items():
            a.instruction = a.instruction.replace("{" + old, "{" + new)
    if renames and kind != "mas":
        def walk(node):
            if node.type == "agent" and node.agent_name in renames:
                node.agent_name = renames[node.agent_name]
            for c in node.children or []:
                walk(c)
        walk(config.pipeline)
    return config


def fmt_seconds(sec: float) -> str:
    """Время в том же виде, в каком его показывает интерфейс."""
    if sec < 60:
        return f"{sec:.1f} с"
    return f"{int(sec // 60)} мин {round(sec % 60):02d} с"


def shorten(text: str, limit: int = 340) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    cut = text[:limit].rfind(". ")
    return text[: cut + 1] if cut > limit * 0.5 else text[: limit - 1] + "…"


def build_preset(args, config, steps, state, totals, extra_tools=None, extra_tokens=None) -> dict:
    """Собирает сценарий в формате presets.js."""
    agents = list(getattr(config, "agents", None) or ([config.coordinator] + list(config.workers)))
    outputs = {a.name: getattr(a, "output_key", None) for a in agents}
    agent_tools = {a.name: list(a.tools or []) for a in agents}
    # шаги, начавшиеся одновременно, помечаются как одна parallel-ветка
    groups: dict[str, str] = {}
    for i, s in enumerate(steps):
        same = [t for t in steps if abs(t["started_at"] - s["started_at"]) < 0.5]
        if len(same) > 1:
            groups[s["agent"]] = f"p{min(steps.index(t) for t in same)}"

    steps = list(steps)
    routed = [a for a, calls in (extra_tools or {}).items()
              if "transfer_to_agent" in calls and all(st["agent"] != a for st in steps)]
    for name in routed:                     # координатор в MAS: вызвал transfer_to_agent и передал ход
        steps.insert(0, {"agent": name, "ms": 1200, "started_at": 0.0,
                         "tokens": (extra_tokens or {}).get(name, 0),
                         "tools": ["transfer_to_agent"]})

    trace = []
    for i, s in enumerate(steps):
        ev = {"agent": s["agent"], "phase": "шаг агента"}
        if s["agent"] in groups:
            ev["group"] = groups[s["agent"]]
        ev["tokens"] = s["tokens"]
        ev["ms"] = s["ms"]
        text = state.get(outputs.get(s["agent"], ""), "")
        calls = list(s["tools"])
        if calls:
            # ADK отдаёт имя функции внутри MCP-сервера («sequentialthinking»),
            # а в конфиге у агента указан сам сервер («sequential-thinking») —
            # на бейдже показываем сервер, имя функции уходит в пояснение.
            servers = agent_tools.get(s["agent"], [])
            # у агента может быть несколько MCP-серверов: сопоставляем функцию с её сервером
            fn_owner = {"execute": "sandbox-light", "repl": "sandbox-light",
                        "sequentialthinking": "sequential-thinking"}
            owner = fn_owner.get(calls[0])
            if owner and owner in servers:
                servers = [owner]
            if calls[0] == "transfer_to_agent":
                ev["phase"] = "маршрутизация"
                ev["tool"] = "transfer_to_agent"
                ev["toolNote"] = "передача профильному агенту"
            elif calls[0] == "exit_loop":
                ev["tool"] = "exit_loop"
                ev["toolNote"] = "критерии выполнены"
            else:
                ev["tool"] = servers[0] if len(servers) == 1 else calls[0]
                names = ", ".join(sorted(set(c for c in calls if c != "exit_loop")))
                ev["toolNote"] = f"{names}, вызовов: {len(calls)}" if len(calls) > 1 else names
        ev["text"] = shorten(text) or (
            "Определил тему обращения и передал его профильному агенту."
            if ev.get("tool") == "transfer_to_agent" else "(агент завершил шаг без текстового вывода)")
        if i == len(steps) - 1:
            ev["final"] = True
        trace.append(ev)

    total_tokens = sum(e["tokens"] for e in trace)
    elapsed = totals.get("elapsed") or 0.0
    auto = f"{fmt_seconds(elapsed)} · {total_tokens / 1000:.1f}к токенов".replace(".", ",")

    return {
        "id": args.id,
        "title": args.title,
        "domain": args.domain,
        "kind": args.kind,
        "model": agents[0].model or args.model,
        "query": args.query,          # полностью: сценарий должен перезапускаться тем же запросом
        # Постановка, по которой мета-агент собрал систему. Раньше здесь стояло `task` —
        # переменная из main(), в build_preset её нет, и запись прогона падала NameError.
        "brief": args.task,
        "summary": args.summary
        or "Конфигурация сгенерирована мета-агентом FEDOT.MAS и исполнена на самом деле: "
        "тексты, токены и время — из фактического запуска.",
        "manual": args.manual,
        "auto": auto,
        "genSteps": [
            "Разбор постановки задачи",
            "Этап 1: генерация пула агентов и подбор инструментов",
            "Этап 2: сборка дерева pipeline",
            "Валидация MAWConfig: ссылки на агентов и переменные состояния",
        ],
        "config": json.loads(config.model_dump_json()),
        "trace": trace,
    }


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True, help="постановка задачи для мета-агента")
    ap.add_argument("--query", required=True, help="запрос, на котором исполняется система")
    ap.add_argument("--model", default="openai/gpt-oss-120b", help="модель агентов системы")
    ap.add_argument("--meta-model", default=None,
                    help="модель мета-агента (по умолчанию та же); полезно, если открытая модель "
                         "нестабильно отдаёт структурированный конфиг")
    ap.add_argument("--tools", nargs="*", default=["sandbox-light", "sequential-thinking"])
    ap.add_argument("--id", default="real_run")
    ap.add_argument("--title", default="Реальный прогон")
    ap.add_argument("--domain", default="live")
    ap.add_argument("--summary", default="")
    ap.add_argument("--manual", default="оценка не проводилась")
    ap.add_argument("--kind", default="maw", choices=["maw", "mas"])
    ap.add_argument("--sources", default="[]",
                    help='JSON-список источников данных: [{"title": "...", "note": "...", "url": "..."}]')
    ap.add_argument("--no-compare", action="store_true",
                    help="не запускать сравнение с одной моделью и судью")
    ap.add_argument("--judge-model", default="google/gemini-2.5-flash")
    ap.add_argument("--max-output-tokens", type=int, default=4000,
                    help="лимит ответа агента; без него провайдер может обрезать длинный ответ "
                         "и запуск падает с MAX_TOKENS")
    ap.add_argument("--out", default="preset.js")
    args = ap.parse_args()

    tracer = TraceCollector()
    cls = MAS if args.kind == "mas" else MAW
    system = cls(
        meta_model=args.meta_model or args.model,
        worker_models=[args.model],
        mcp_servers=args.tools,
        plugins=[LoggingPlugin(), tracer],
    )
    task = args.task + (
        "\n\nВажно: имена агентов, их описания и инструкции — на русском языке. "
        "Имена агентов пиши строчными буквами через подчёркивание."
    )

    gen_t0 = time.monotonic()
    config = sanitize(await system.generate_config(task), args.kind)
    # Мета-агент MAS вписывает координатору исполнителей в «инструменты», а это не MCP-серверы:
    # сборка падала на Unknown MCP server. Сервер стенда такие ссылки чистит так же.
    for agent in (list(getattr(config, "agents", None) or [])
                  or [config.coordinator] + list(config.workers)):
        unknown = [t for t in (agent.tools or []) if t not in args.tools]
        if unknown:
            print(f"У агента {agent.name} убраны неизвестные инструменты: {unknown}")
            agent.tools = [t for t in agent.tools if t in args.tools]
    if args.max_output_tokens and hasattr(config, "agents"):
        for agent in config.agents:
            agent.max_output_tokens = args.max_output_tokens
    gen_steps = tracer.take()          # шаги мета-агента в журнал запуска не попадают
    gen_tokens = sum(s["tokens"] for s in gen_steps)
    gen_elapsed = time.monotonic() - gen_t0
    print("Сгенерированная система:\n", config, sep="")
    print(f"Генерация: {gen_elapsed:.1f} с, {gen_tokens} токенов")

    agents_all = list(getattr(config, "agents", None) or ([config.coordinator] + list(config.workers)))
    run_config, names = config, AgentNames()
    if args.kind == "mas":
        # Исполнители MAS становятся инструментами координатора, а OpenAI не принимает
        # кириллицу в имени инструмента: запускаем копию с латинскими именами.
        run_config = config.model_copy(deep=True)
        names = latinize_mas(run_config)
        for worker, run_worker in zip(config.workers, run_config.workers):
            worker.output_key = run_worker.output_key     # «<имя>_output» по исходному имени
    outputs_by_agent = {a.name: getattr(a, "output_key", None) for a in agents_all}
    result = await system.build_and_run(run_config, args.query)
    state = dict(result if isinstance(result, dict) else getattr(result, "state", {}))
    if names.shown:
        # В сценарий — исходные имена; вызов исполнителя MAS для интерфейса — передача хода
        def call(tool: str) -> str:
            return "transfer_to_agent" if names.is_agent(tool) else tool

        state = {k: names.text(v) if isinstance(v, str) else v for k, v in state.items()}
        for step in tracer.steps:
            step["agent"] = names.name(step["agent"])
            step["tools"] = [call(t) for t in step["tools"]]
        tracer._tools = {names.name(a): [call(t) for t in calls] for a, calls in tracer._tools.items()}
        tracer._tokens = {names.name(a): n for a, n in tracer._tokens.items()}
    last = system.last_result
    totals = {
        "prompt": getattr(last, "total_prompt_tokens", 0),
        "completion": getattr(last, "total_completion_tokens", 0),
        "elapsed": getattr(last, "elapsed", 0.0),
    }

    # Итог системы — артефакт последнего содержательного агента, а не отзыв критика:
    # критик пишет «план полный и обоснованный», и это не ответ на задачу пользователя.
    critic_re = re.compile(r"критик|critic|валид|valid|провер|review|judge|качеств|контрол|аудит|реценз|quality", re.I)
    answer = ""
    for step in reversed(tracer.steps):
        if critic_re.search(step["agent"]):
            continue
        key = outputs_by_agent.get(step["agent"])
        if key and isinstance(state.get(key), str) and state[key].strip():
            answer = state[key].strip()
            break
    # Последний агент иногда отдаёт короткую сводку вместо расчёта — берём содержательный артефакт.
    if answer:
        pool = [state[k] for a, k in outputs_by_agent.items()
                if not critic_re.search(a) and isinstance(state.get(k), str)]
        longest = max(pool, key=len, default="")
        if len(longest) > len(answer) * 2:
            answer = longest.strip()

    if not answer:  # запасной путь: любой непустой артефакт
        for key in reversed(list(state)):
            if key != "user_query" and isinstance(state.get(key), str) and state[key].strip():
                answer = state[key].strip()
                break

    preset = build_preset(args, config, tracer.steps, state, totals,
                          extra_tools=tracer._tools, extra_tokens=tracer._tokens)
    preset["answer"] = answer
    preset["answerMeta"] = (
        f"{'MASConfig' if args.kind == 'mas' else 'MAWConfig'} · "
        f"{config.agents[0].model if args.kind == 'maw' else config.coordinator.model} · "
        f"{totals['prompt'] + totals['completion']} токенов · {totals['elapsed']:.1f} с"
    )
    preset["real"] = True
    preset["kind"] = args.kind
    preset["sources"] = json.loads(args.sources)

    if not args.no_compare and answer:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(base_url=os.getenv("OPENAI_BASE_URL"), api_key=os.getenv("OPENAI_API_KEY"))
        print("\nСравнение: та же задача одной моделью…")
        t0 = time.monotonic()
        single = await client.chat.completions.create(
            model=args.model,
            messages=[
                {"role": "system", "content": "Ты эксперт-аналитик. Отвечай по-русски, по существу и структурировано."},
                {"role": "user", "content": args.query},
            ],
        )
        single_text = single.choices[0].message.content or ""
        preset["baseline"] = {
            "answer": single_text,
            "model": args.model,
            "tokens": (single.usage.prompt_tokens or 0) + (single.usage.completion_tokens or 0),
            "seconds": round(time.monotonic() - t0, 1),
        }
        print(f"  {preset['baseline']['tokens']} токенов, {preset['baseline']['seconds']} с")

        print("Судья сравнивает ответы…")
        verdict = await client.chat.completions.create(
            model=args.judge_model,
            messages=[{"role": "user", "content": JUDGE_PROMPT.format(
                query=args.query, a=answer, b=single_text)}],
        )
        vtext = (verdict.choices[0].message.content or "").strip()
        preset["judge"] = {"verdict": vtext, "winner": _winner(vtext), "model": args.judge_model}
        print(f"  победитель: {preset['judge']['winner']}")
    preset["gen"] = (
        f"{fmt_seconds(gen_elapsed)} · {gen_tokens / 1000:.1f}к токенов".replace(".", ",")
    )
    with open(args.out, "w") as f:
        f.write(json.dumps(preset, ensure_ascii=False, indent=2))
    print(f"\nСценарий записан в {args.out}")
    print(f"шагов: {len(preset['trace'])} | генерация: {preset['gen']} | запуск: {preset['auto']}")
    print("Вставьте содержимое файла как элемент массива window.PRESETS в presets.js")


if __name__ == "__main__":
    asyncio.run(main())
