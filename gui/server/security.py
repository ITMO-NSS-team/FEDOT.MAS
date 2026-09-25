"""Доступ к стенду: токен в ссылке и ключ провайдера от пользователя.

Два независимых заслона. Токен решает, кого вообще пускать к API, ключ — чьими
деньгами оплачиваются вызовы моделей. Подробности в README, раздел о безопасности.
"""

from __future__ import annotations

import json
import os
import secrets

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fedotmas.common.codex_cli import is_codex_model
from fedotmas.common.logging import get_logger

from .config import ACCESS_TOKEN, DEFAULT_MODEL, MODELS, PUBLIC_MODE
from .schemas import KeyIn

_log = get_logger("gui.security")

# Открыто без токена: страница должна загрузиться, чтобы показать форму ввода ключа.
OPEN_API = {"/api/status"}
# Пути, которые тратят деньги на модели: без ключа пользователя их не пускаем.
NEEDS_KEY = {"/api/generate", "/api/generate_stream", "/api/run", "/api/prepare",
             "/api/baseline", "/api/effort", "/api/effort_breakdown",
             "/api/judge", "/api/judge_stream"}

# Ключ пользователя живёт в памяти процесса и подставляется в окружение — оттуда его
# читают и FEDOT.MAS (resolve_model_config), и прямые вызовы клиента. Расчёт на одного
# пользователя: стенд демонстрационный.
_user_key_set = False


def user_key_set() -> bool:
    """Ввёл ли пользователь свой ключ. Значение меняется, поэтому отдаём функцией."""
    return _user_key_set


def mask(key: str) -> str:
    """Ключ нигде не должен появляться целиком — ни в логах, ни в ответах API."""
    key = key.strip()
    return f"{key[:6]}…{key[-4:]}" if len(key) > 14 else "…"


def _key_available(model: str) -> bool:
    """Есть ли ключ, которым реально пойдёт запрос этой модели.

    litellm читает для openrouter/* только OPENROUTER_API_KEY — наличие
    OPENAI_API_KEY такую модель не оплатит, и наоборот.
    """
    if model.startswith("openrouter/"):
        return bool(os.getenv("OPENROUTER_API_KEY"))
    return bool(os.getenv("OPENAI_API_KEY"))


# Имена, под которыми стенд открывают штатно. Туннель добавляет своё — его имя
# берётся из GUI_ALLOWED_HOSTS (через запятую), иначе запрос с него отклонится.
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "[::1]", "0.0.0.0"}
_EXTRA_HOSTS = {h.strip().lower() for h in os.getenv("GUI_ALLOWED_HOSTS", "").split(",") if h.strip()}


def _host_allowed(host: str) -> bool:
    """Свой ли это адрес. Пустой Host считаем своим: его не ставят только не-браузеры."""
    if not host:
        return True
    host = host.strip().rstrip(".").strip("[]")     # «localhost.», «[::1]» — тот же адрес
    if host in _LOCAL_HOSTS or host in _EXTRA_HOSTS:
        return True
    # Туннели выдают имя при каждом запуске, заранее его не знаешь. В публичном режиме
    # заслоном служит токен доступа, который межсайтовый запрос поставить не может.
    return PUBLIC_MODE


def install(app: FastAPI) -> None:
    """Вешает на приложение проверку доступа и запрет кеширования."""

    @app.middleware("http")
    async def guard(request: Request, call_next):
        """Единая проверка доступа: один заслон вместо проверки в каждом обработчике."""
        path = request.url.path
        if not path.startswith("/api/"):
            return await call_next(request)
        # Чужая страница, открытая в браузере владельца, может послать сюда простой
        # POST (тело Blob без Content-Type — предполётного запроса не будет) и
        # запустить агентов за его счёт. Ответ ей не виден, но действие произойдёт.
        # Сверяем и Host, и Origin со списком ожидаемых имён, а не друг с другом:
        # сравнение «origin == host» проходило при DNS rebinding — злой домен,
        # переведённый на 127.0.0.1, присылает совпадающую пару и попадает внутрь.
        if request.method != "GET":
            raw_host = (request.headers.get("host") or "").strip().lower()
            # У IPv6 адрес в скобках, и разрезать по первому двоеточию нельзя
            if raw_host.startswith("["):
                host, _, port_text = raw_host.partition("]")
                host, port_text = host + "]", port_text.lstrip(":")
            else:
                host, _, port_text = raw_host.partition(":")
            host_port = int(port_text) if port_text.isdigit() else None
            if host and not _host_allowed(host):
                _log.warning("Запрос с неожиданным Host отклонён | host={}", host)
                return JSONResponse({"error": "неожиданное имя хоста"}, status_code=403)
            origin = request.headers.get("origin")
            if origin:
                from urllib.parse import urlparse

                # «null» присылает песочница iframe, srcdoc и страница из data:. Раньше
                # у него не было имени хоста, пустое имя считалось своим — и заслон
                # обходился именно тем способом, от которого он и ставился.
                try:
                    parsed = urlparse(origin)
                    origin_host, origin_port = (parsed.hostname or "").lower(), parsed.port
                except ValueError:                  # в том числе порт вне диапазона
                    origin_host, origin_port = "", None
                if not origin_host or not _host_allowed(origin_host):
                    _log.warning("Межсайтовый запрос отклонён | origin={}", origin)
                    return JSONResponse({"error": "запрос пришёл со стороннего сайта"},
                                        status_code=403)
                # Имя своё, а порт другой — это другой сайт: страница Jupyter или чужого
                # dev-сервера на localhost:8888. Проверено запросом: без этой сверки она
                # запускала агентов, которым в локальном режиме доступны файлы машины.
                # В публичном режиме такой запрос и так остановит токен доступа.
                if (not PUBLIC_MODE and origin_host.rstrip(".") in _LOCAL_HOSTS
                        and origin_port != host_port):
                    _log.warning("Запрос со страницы на другом порту отклонён | origin={}", origin)
                    return JSONResponse({"error": "запрос пришёл со стороннего сайта"},
                                        status_code=403)
        if PUBLIC_MODE and path not in OPEN_API:
            if not secrets.compare_digest(request.headers.get("x-access-token", ""), ACCESS_TOKEN):
                return JSONResponse({"error": "нет доступа: откройте ссылку целиком, вместе с токеном"},
                                    status_code=401)
            if path == "/api/export-presets":
                # Пишет файл на диск рядом с кодом — наружу такое не отдаём.
                return JSONResponse({"error": "экспорт пресетов доступен только локально"},
                                    status_code=403)
        if path in NEEDS_KEY:
            selected_model = DEFAULT_MODEL
            try:
                payload = json.loads((await request.body()) or b"{}")
                selected_model = payload.get("model") or DEFAULT_MODEL
            except (json.JSONDecodeError, AttributeError, UnicodeDecodeError):
                pass
            if not is_codex_model(selected_model) and not _key_available(selected_model):
                return JSONResponse({"error": "не задан ключ провайдера"}, status_code=428)
        return await call_next(request)

    @app.middleware("http")
    async def no_cache(request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store, max-age=0"
        return response


async def set_key(body: KeyIn) -> dict:
    """Принимает ключ провайдера от пользователя и проверяет его живым запросом.

    Проверка нужна, чтобы опечатка в ключе всплыла сразу, а не посреди прогона
    на демонстрации. Пустой ключ означает выход: ключ забывается.
    """
    global _user_key_set
    key = (body.key or "").strip()
    if not key:
        os.environ.pop("OPENAI_API_KEY", None)
        os.environ.pop("OPENROUTER_API_KEY", None)
        _user_key_set = False
        return {"ok": True, "has_key": False}
    # Ключи OpenRouter начинаются с sk-or-: проверяем их на самом OpenRouter,
    # моделью из списка стенда — DEFAULT_MODEL может оказаться codex-моделью,
    # которой у провайдера нет.
    openrouter = key.startswith("sk-or-")
    if openrouter:
        base = "https://openrouter.ai/api/v1"
        probe = next((m["id"].removeprefix("openrouter/") for m in MODELS
                      if m["id"].startswith("openrouter/")), "openai/gpt-4o")
    else:
        base = (os.getenv("OPENAI_BASE_URL") or "").strip()
        probe = DEFAULT_MODEL
    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(base_url=base or None, api_key=key, timeout=25)
        await client.chat.completions.create(
            model=probe, max_tokens=1,
            messages=[{"role": "user", "content": "ping"}])
    except Exception as exc:
        # В тексте ошибки провайдер иногда повторяет присланный ключ — вычищаем.
        note = str(exc).replace(key, mask(key))[:300]
        _log.warning("Ключ не принят: {}", note)
        # Сырой ответ провайдера на демонстрации читать некому — переводим частые случаи.
        if "401" in note or "not found" in note.lower() or "invalid" in note.lower():
            human = "провайдер не признал этот ключ"
        elif "402" in note or "credit" in note.lower() or "quota" in note.lower():
            human = "на ключе не осталось средств"
        elif "429" in note:
            human = "провайдер ограничил частоту запросов, попробуйте через минуту"
        elif "timeout" in note.lower() or "connect" in note.lower():
            human = "не удалось достучаться до провайдера"
        else:
            human = note
        return {"ok": False, "error": human}
    os.environ["OPENAI_API_KEY"] = key
    if openrouter:
        # litellm оплачивает openrouter/* только из этой переменной
        os.environ["OPENROUTER_API_KEY"] = key
    _user_key_set = True
    _log.info("Принят ключ пользователя {}", mask(key))
    return {"ok": True, "has_key": True, "masked": mask(key)}
