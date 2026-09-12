"""Доступ к стенду: токен в ссылке и ключ провайдера от пользователя.

Два независимых заслона. Токен решает, кого вообще пускать к API, ключ — чьими
деньгами оплачиваются вызовы моделей. Подробности в README, раздел о безопасности.
"""

from __future__ import annotations

import os
import secrets

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fedotmas.common.logging import get_logger

from .config import ACCESS_TOKEN, DEFAULT_MODEL, PUBLIC_MODE
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
        # Браузер обязан проставить Origin для межсайтового запроса — на этом и ловим.
        # У curl и прочих не-браузеров заголовка нет, их не трогаем.
        origin = request.headers.get("origin")
        if origin and request.method != "GET":
            host = request.headers.get("host", "")
            if origin.split("://")[-1] != host:
                _log.warning("Межсайтовый запрос отклонён | origin={} host={}", origin, host)
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
        if path in NEEDS_KEY and not os.getenv("OPENAI_API_KEY"):
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
        _user_key_set = False
        return {"ok": True, "has_key": False}
    base = (os.getenv("OPENAI_BASE_URL") or "").strip()
    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(base_url=base or None, api_key=key, timeout=25)
        await client.chat.completions.create(
            model=DEFAULT_MODEL, max_tokens=1,
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
    _user_key_set = True
    _log.info("Принят ключ пользователя {}", mask(key))
    return {"ok": True, "has_key": True, "masked": mask(key)}
