"""Веб-приложение стенда: маршруты API и раздача интерфейса.

Логика по модулям: config — настройки, security — доступ, normalize — правка
конфигурации от мета-агента, judge — сравнение ответов, streaming — поток событий.
Здесь остались только сами маршруты.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import tempfile
import time
import uuid
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fedotmas import MAS, MAW, MASConfig, MAWConfig
from fedotmas.common.logging import get_logger
from fedotmas.mcp import get_server_descriptions, resolve_mcp_registry
from fedotmas.plugins import LoggingPlugin

from . import security
from .config import (AGENT_MAX_OUTPUT_TOKENS, DEFAULT_MODEL, GENERATE_ATTEMPTS, JUDGE_MODEL, MODELS,
                     PUBLIC_MODE, RU_HINT, SAFE_TOOLS, SCRAPING, SERVER_RUN_ID,
                     SMITHERY_API_KEY, SMITHERY_DETAIL, SMITHERY_SEARCH, STATIC_DIR,
                     WEB_SEARCH)
from .judge import _judge_impl
from .llm import client as _client
from .normalize import (_ensure_calculator, _ensure_data_tools, _ensure_dependent_after_parallel,
                        _ensure_lookup_tools, _ensure_web_tool, _mcp_registry_for,
                        _tools_hint, _with_data_source, sanitize_config)
from .prompts import (BREAKDOWN_PROMPT, EFFORT_PROMPT, PREPARE_PROMPT, RETRY_HINT,
                      TOOL_DESCRIPTIONS)
from .schemas import (BaselineIn, EffortIn, ExportIn, GenerateIn, JudgeIn,
                      PrepareIn, RunIn)
from .streaming import StreamPlugin, sse_stream

_log = get_logger("gui.live")


def _allowed_tools(asked: list[str] | None) -> list[str]:
    """Из присланного клиентом списка оставляем только разрешённые на этой машине.

    SAFE_TOOLS — не подсказка интерфейсу, а граница: без пересечения клиент мог бы
    включить себе инструменты, которые владелец стенда сознательно не открывал
    (browser-usage, полную песочницу), просто перечислив их в запросе.
    """
    if asked is None:
        return list(SAFE_TOOLS)
    allowed = [t for t in asked if t in SAFE_TOOLS]
    dropped = sorted(set(asked) - set(allowed))
    if dropped:
        _log.warning("Запрошены недоступные инструменты, отброшены: {}", dropped)
    return allowed

app = FastAPI(title="FEDOT.MAS GUI live")
security.install(app)
app.post("/api/key")(security.set_key)


@app.get("/api/status")
async def status() -> dict:
    registry = resolve_mcp_registry("all") or {}
    return {
        "live": True,
        "model": DEFAULT_MODEL,
        "models": MODELS,
        "judge_model": JUDGE_MODEL,
        "safe_tools": SAFE_TOOLS,
        # подписи для списка инструментов в окне создания сценария
        "tools": [{"id": t, "note": TOOL_DESCRIPTIONS.get(t, "").split(" — ")[-1]}
                  for t in SAFE_TOOLS],
        "web_search": WEB_SEARCH,
        "scraping": SCRAPING,
        "mcp_servers": sorted(registry),
        "run_id": SERVER_RUN_ID,
        "has_key": bool(os.getenv("OPENAI_API_KEY")),
        "base_url": os.getenv("OPENAI_BASE_URL", ""),
        "public": PUBLIC_MODE,
        "user_key": security.user_key_set(),
    }



async def _generate_impl(body: GenerateIn, queue: asyncio.Queue) -> dict:
    """Собственно генерация конфигурации; события мета-агента уходят в очередь."""
    model = body.model or DEFAULT_MODEL
    tools = _allowed_tools(body.tools)
    stream = StreamPlugin(queue)
    servers, custom_names = _mcp_registry_for(tools, body.custom_mcp)
    cls = MAS if body.kind == "mas" else MAW
    system = cls(meta_model=model, worker_models=[model],
                 mcp_servers=servers,
                 plugins=[LoggingPlugin(), stream])

    task = body.task + (RU_HINT if body.russian else "")
    t0 = time.monotonic()
    config = None
    last_error: Exception | None = None
    for attempt in range(GENERATE_ATTEMPTS):
        # Мета-агент временами ссылается в пайплайне на агента, которого нет в пуле,
        # или отдаёт невалидную схему — на демо это роняло создание сценария целиком.
        hint = "" if attempt == 0 else RETRY_HINT.format(error=last_error)
        try:
            config = await system.generate_config(task + hint)
            break
        except Exception as exc:
            last_error = exc
            _log.warning("Попытка {}/{} генерации не удалась | {}: {}",
                         attempt + 1, GENERATE_ATTEMPTS, type(exc).__name__, exc)
    if config is None:
        return {"ok": False, "error": f"{type(last_error).__name__}: {last_error}"}

    config = sanitize_config(config, body.kind, custom_names)
    if body.web and WEB_SEARCH:
        _ensure_web_tool(config, body.kind)
    _ensure_data_tools(config, body.kind, f"{body.task} {body.query or ''}")
    _ensure_dependent_after_parallel(config, body.kind)
    _ensure_calculator(config, body.kind)
    _ensure_lookup_tools(config, body.kind, f"{body.task} {body.query or ''}")
    for agent in getattr(config, "agents", []) or []:
        agent.max_output_tokens = AGENT_MAX_OUTPUT_TOKENS

    return {
        "ok": True,
        "kind": body.kind,
        "config": json.loads(config.model_dump_json()),
        "gen": {"tokens": stream.tokens, "seconds": round(time.monotonic() - t0, 1)},
        "model": model,
    }


@app.post("/api/generate")
async def generate(body: GenerateIn) -> dict:
    return await _generate_impl(body, asyncio.Queue())


@app.post("/api/generate_stream")
async def generate_stream(body: GenerateIn) -> StreamingResponse:
    """То же самое, но с потоком событий: без него генерация выглядит как зависание."""
    queue: asyncio.Queue = asyncio.Queue()

    async def execute() -> None:
        try:
            queue.put_nowait({"type": "done", **(await _generate_impl(body, queue))})
        except Exception as exc:
            queue.put_nowait({"type": "done", "ok": False, "error": f"{type(exc).__name__}: {exc}"})
        finally:
            queue.put_nowait(None)
    return sse_stream(queue, execute)


@app.post("/api/run")
async def run(body: RunIn) -> StreamingResponse:
    tools = _allowed_tools(body.tools)
    model = body.model or DEFAULT_MODEL
    servers, custom_names = _mcp_registry_for(tools, body.custom_mcp)
    queue: asyncio.Queue = asyncio.Queue()
    stream = StreamPlugin(queue)
    is_mas = body.kind == "mas"
    cls = MAS if is_mas else MAW
    system = cls(worker_models=[model], mcp_servers=servers, plugins=[LoggingPlugin(), stream])
    config = MASConfig(**body.config) if is_mas else MAWConfig(**body.config)
    config = sanitize_config(config, body.kind, custom_names)   # может прийти из файла

    # У агента в конфигурации своя модель, и она сильнее worker_models: без явной
    # перезаписи выбор модели запуска не влиял бы на сохранённые сценарии.
    if body.model:
        picked = (getattr(config, "agents", None)
                  or [config.coordinator] + list(config.workers))
        for agent in picked:
            agent.model = model


    async def execute() -> None:
        t0 = time.monotonic()
        try:
            result = await system.build_and_run(config, body.query)
            state = dict(result if isinstance(result, dict) else getattr(result, "state", {}))
            last = system.last_result
            queue.put_nowait({
                "type": "done",
                "elapsed": round(getattr(last, "elapsed", time.monotonic() - t0), 1),
                "tokens": (getattr(last, "total_prompt_tokens", 0) or 0)
                + (getattr(last, "total_completion_tokens", 0) or 0),
                # ответ уходит в сравнение и судье целиком: обрезка искажала бы оценку
                "state": {k: str(v)[:40000] for k, v in state.items() if k != "user_query"},
            })
        except Exception as exc:
            queue.put_nowait({"type": "error", "error": f"{type(exc).__name__}: {exc}"})
        finally:
            queue.put_nowait(None)
    return sse_stream(queue, execute)


@app.post("/api/prepare")
async def prepare(body: PrepareIn) -> dict:
    """Делит свободный текст пользователя на постановку для мета-агента и запрос для запуска."""
    client, model = _client(body.model or DEFAULT_MODEL)
    hint = ("пайплайн агентов с параллельными ветками и циклами"
            if body.kind == "maw" else "координатор, маршрутизирующий задачи специалистам")
    try:
        resp = await client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": PREPARE_PROMPT.format(
                text=body.text, kind=body.kind, kind_hint=hint,
                tools_hint=_tools_hint(body.web, body.tools))}],
        )
        data = json.loads(resp.choices[0].message.content or "{}")
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    task, query = (data.get("task") or "").strip(), (data.get("query") or "").strip()
    if not task:
        return {"ok": False, "error": "не удалось выделить постановку из текста"}
    query = _with_data_source(query or body.text.strip(), task)
    return {"ok": True, "task": task, "query": query, "model": model}


@app.post("/api/baseline")
async def baseline(body: BaselineIn) -> dict:
    """Тот же запрос, но решает одна модель без мультиагентной системы."""
    client, model = _client(body.model or DEFAULT_MODEL)
    t0 = time.monotonic()
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Ты эксперт-аналитик. Отвечай по-русски, по существу и структурировано."},
                {"role": "user", "content": body.query},
            ],
        )
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    usage = resp.usage
    return {
        "ok": True,
        "answer": resp.choices[0].message.content or "",
        "model": model,
        "tokens": (getattr(usage, "prompt_tokens", 0) or 0) + (getattr(usage, "completion_tokens", 0) or 0),
        "seconds": round(time.monotonic() - t0, 1),
    }


@app.post("/api/effort")
async def effort(body: EffortIn) -> dict:
    """Оценка трудоёмкости ручной разработки такой же системы — считает LLM по составу конфигурации."""
    client, model = _client(body.model or DEFAULT_MODEL)
    agents = []
    for agent in (body.config or {}).get("agents") or []:
        agents.append(f"- {agent.get('name')}: инструменты {', '.join(agent.get('tools') or []) or 'нет'}")
    if not agents and body.config:                 # у MASConfig состав описан иначе
        coord = (body.config.get("coordinator") or {}).get("name")
        agents = [f"- координатор {coord}"] + [
            f"- {w.get('name')}: инструменты {', '.join(w.get('tools') or []) or 'нет'}"
            for w in body.config.get("workers") or []]
    pipeline = json.dumps((body.config or {}).get("pipeline"), ensure_ascii=False)[:1500]
    summary = "\n".join(agents) + ("\n\nПайплайн: " + pipeline if pipeline != "null" else "")

    try:
        resp = await client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": EFFORT_PROMPT.format(
                task=body.task[:4000], config=summary or "состав неизвестен")}],
        )
        data = json.loads(resp.choices[0].message.content or "{}")
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    estimate = (data.get("estimate") or "").strip()[:24]
    if not estimate:
        return {"ok": False, "error": "модель не вернула оценку"}
    return {"ok": True, "estimate": estimate,
            "detail": (data.get("detail") or "").strip(), "model": model}


@app.post("/api/export-presets")
async def export_presets(body: ExportIn) -> dict:
    """Сохраняет свои сценарии из браузера в файл, чтобы вшить их в автономную копию.

    Сценарии, созданные через форму, живут в localStorage и в сборку не попадают.
    Эта ручка выгружает их рядом с presets.js, откуда их подхватывает build_offline.py.
    """
    payload = json.dumps(body.presets, ensure_ascii=False)
    target = STATIC_DIR / "presets_custom.js"
    target.write_text(
        "/* Свои сценарии, выгруженные из браузера: записи прогонов для автономной копии. */\n"
        "window.PRESETS = (window.PRESETS || []).concat(\n" + payload + "\n);\n",
        encoding="utf-8")
    _log.info("Сценарии выгружены | штук={} файл={}", len(body.presets), target.name)
    return {"ok": True, "count": len(body.presets), "file": target.name,
            "bytes": target.stat().st_size}


@app.post("/api/effort_breakdown")
async def effort_breakdown(body: EffortIn) -> dict:
    """Декомпозиция задачи на подзадачи с оценкой времени по каждой.

    Общая оценка «≈ N чел.-дней» ничего не объясняет; разбор показывает, из чего
    складывается ручная трудоёмкость и что именно берёт на себя система.
    """
    client, model = _client(body.model or DEFAULT_MODEL)
    agents = []
    for agent in (body.config or {}).get("agents") or []:
        agents.append(f"- {agent.get('name')}: {', '.join(agent.get('tools') or []) or 'без инструментов'}")
    summary = "\n".join(agents) or "состав неизвестен"

    try:
        resp = await client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": BREAKDOWN_PROMPT.format(
                task=body.task[:6000], config=summary)}],
        )
        data = json.loads(resp.choices[0].message.content or "{}")
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    items = []
    for raw in (data.get("subtasks") or [])[:10]:
        try:
            hours = float(raw.get("hours") or 0)
        except (TypeError, ValueError):
            hours = 0.0
        name = str(raw.get("name") or "").strip()
        if not name or hours <= 0:
            continue
        items.append({"name": name[:160], "hours": round(hours, 2),
                      "note": str(raw.get("note") or "").strip()[:300]})
    if not items:
        return {"ok": False, "error": "не удалось разложить задачу на подзадачи"}

    total = round(sum(i["hours"] for i in items), 2)
    # Человеко-дни считаем по восьмичасовому рабочему дню — так их и подписывает
    # интерфейс. Сумма часов остаётся главной цифрой, дни — это перевод для наглядности.
    return {"ok": True, "subtasks": items, "total_hours": total,
            "total_days": round(total / 8, 1), "model": model}


# Файл как источник данных. Ссылку агент скачивает сам, а лежащий на компьютере
# файл иначе до него не доходит: кладём его в отдельный каталог и передаём путь в
# запросе — дальше его читает инструмент «document», тот же, что и скачанные файлы.
UPLOAD_MAX_BYTES = int(os.getenv("GUI_UPLOAD_MAX_MB", "25")) * 1024 * 1024
# Каталог загрузок — свой на каждый запуск сервера и только для владельца (0700).
# Предсказуемый путь в общем /tmp читался бы любым локальным пользователем, а его
# можно было бы заранее подменить симлинком: mkdir(exist_ok=True) владельца не проверяет.
UPLOAD_DIR = Path(tempfile.mkdtemp(prefix="fedotmas-gui-uploads-"))


@app.post("/api/upload")
async def upload(request: Request, file: UploadFile = File(...)) -> dict:
    """Принимает файл-источник и возвращает путь, по которому его прочитает агент."""
    if "document" not in SAFE_TOOLS:
        # Без инструмента чтения файл бесполезен: в публичном режиме он отключён.
        return {"ok": False, "error": "чтение файлов на этом стенде отключено "
                                      "(инструмент document недоступен)"}
    # Длину проверяем до чтения: Starlette спулит файл на диск целиком, и предел,
    # применённый после, ограничивал бы только то, что осядет в каталоге загрузок.
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > UPLOAD_MAX_BYTES * 2:
        return {"ok": False, "error": f"файл больше {UPLOAD_MAX_BYTES // 1024 // 1024} МБ"}
    data = await file.read(UPLOAD_MAX_BYTES + 1)
    if len(data) > UPLOAD_MAX_BYTES:
        return {"ok": False, "error": f"файл больше {UPLOAD_MAX_BYTES // 1024 // 1024} МБ"}
    if not data:
        return {"ok": False, "error": "файл пустой"}
    # Имя чистим целиком: в заголовке может приехать и «../», и что угодно ещё.
    # Длину режем — иначе слишком длинное имя роняет запись с OSError и отдаёт 500.
    safe = re.sub(r"[^\w.\- ]+", "_", Path(file.filename or "файл").name).strip()[:120] or "файл"
    target = UPLOAD_DIR / f"{uuid.uuid4().hex[:8]}_{safe}"
    target.write_bytes(data)
    _log.info("Принят файл-источник | имя={} размер={} КБ", safe, len(data) // 1024)
    return {"ok": True, "name": safe, "path": str(target), "size": len(data)}


@app.get("/api/mcp_catalog")
async def mcp_catalog() -> dict:
    """Каталог доступных MCP-серверов: то, что FEDOT.MAS нашёл в mcp-servers/.

    Реестр строится самим FEDOT.MAS (discover_local_servers по [tool.fedotmas.mcp]
    в pyproject каждого сервера), здесь он только раскрывается наружу с пометкой,
    какие серверы включены в живом режиме, а какие требуют ключа или установки.
    """
    registry = resolve_mcp_registry("all") or {}
    descriptions = get_server_descriptions(registry)
    items = []
    for name, cfg in sorted(registry.items()):
        items.append({
            "name": name,
            "description": descriptions.get(name, ""),
            "tags": list(getattr(cfg, "tags", ()) or ()),
            "transport": "http" if hasattr(cfg, "url") else "stdio",
            "enabled": name in SAFE_TOOLS,
            "note": TOOL_DESCRIPTIONS.get(name, ""),
        })
    return {"ok": True, "servers": items, "enabled": SAFE_TOOLS}


@app.get("/api/mcp_search")
async def mcp_search(q: str = "", limit: int = 8) -> dict:
    """Поиск MCP-серверов во внешнем реестре Smithery.

    Локальный каталог — это десяток серверов из mcp-servers/, и под произвольную
    задачу из зала в нём может не оказаться ничего подходящего. В FEDOT.MAS такой
    поиск живёт в ветке feat/meta-agent-mcp-discovery (fedotmas/mcp/pulsemcp.py) и
    в main ещё не влит, поэтому здесь тот же публичный реестр опрашивается напрямую.
    Берём только развёрнутые серверы: у них есть deploymentUrl, который можно сразу
    подключить как HTTP MCP.
    """
    query = (q or "").strip()
    if not query:
        return {"ok": True, "servers": []}
    try:
        import httpx

        async with httpx.AsyncClient(timeout=12.0) as client:
            resp = await client.get(SMITHERY_SEARCH,
                                    params={"q": query, "pageSize": max(limit * 2, 10)})
            resp.raise_for_status()
            found = [s for s in resp.json().get("servers", []) if s.get("isDeployed")][:limit]
            details = await asyncio.gather(
                *[client.get(SMITHERY_DETAIL.format(name=quote(s["qualifiedName"], safe="")))
                  for s in found],
                return_exceptions=True)
    except Exception as exc:
        _log.warning("Поиск в реестре Smithery не удался: {}", str(exc)[:200])
        return {"ok": False, "error": f"реестр не ответил: {type(exc).__name__}", "servers": []}

    servers = []
    for meta, detail in zip(found, details):
        if isinstance(detail, Exception):
            continue
        try:
            url = (detail.json() or {}).get("deploymentUrl") or ""
        except Exception:
            continue
        if not url:
            continue
        servers.append({
            "name": _slugify_mcp(meta["qualifiedName"]),
            "title": meta.get("displayName") or meta["qualifiedName"],
            "description": (meta.get("description") or "")[:220],
            "url": url,
            "uses": meta.get("useCount") or 0,
            "verified": bool(meta.get("verified")),
        })
    _log.info("Реестр Smithery: по запросу «{}» подошло {} серверов", query[:60], len(servers))
    return {"ok": True, "servers": servers, "needs_key": not SMITHERY_API_KEY}


def _slugify_mcp(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:64]


@app.post("/api/judge")
async def judge(body: JudgeIn) -> dict:
    return await _judge_impl(body)


@app.post("/api/judge_stream")
async def judge_stream(body: JudgeIn) -> StreamingResponse:
    """То же сравнение, но с потоком событий: судья работает минутами, и без обратной
    связи интерфейс выглядит зависшим."""
    queue: asyncio.Queue = asyncio.Queue()

    async def execute() -> None:
        try:
            queue.put_nowait({"type": "done", **(await _judge_impl(body, queue))})
        except Exception as exc:
            queue.put_nowait({"type": "done", "ok": False,
                              "error": f"{type(exc).__name__}: {exc}"})
        finally:
            queue.put_nowait(None)
    return sse_stream(queue, execute)


# Статика интерфейса монтируется последней: иначе она перехватила бы /api/.
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="gui")
