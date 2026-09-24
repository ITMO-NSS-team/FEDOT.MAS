"""Проверки внешнего окружения: поиск и браузер для агентов.

Инструмент подключается агентам только если он на машине действительно работает.
Иначе мета-агент выдаёт агенту инструмент, который на первом же вызове падает.
"""

from __future__ import annotations

import time

from fedotmas.common.logging import get_logger

_log = get_logger("gui.infra")


def searxng_alive(url: str, timeout: float = 4) -> bool:
    """Веб-поиск подключаем только если экземпляр SearXNG действительно отвечает."""
    try:
        import urllib.request

        with urllib.request.urlopen(f"{url}/search?q=ping&format=json", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def ensure_searxng(url: str) -> bool:
    """Поднимает остановленный контейнер SearXNG, если он есть на машине."""
    if searxng_alive(url):
        return True
    import os

    # Вызов идёт на импорте server.config: сломанный контейнер задерживал бы
    # каждый старт сервера на десятки секунд. GUI_SEARXNG_AUTOSTART=0 отключает
    # попытку подъёма — стенд стартует сразу, просто без веб-поиска.
    if os.getenv("GUI_SEARXNG_AUTOSTART", "1") == "0":
        _log.info("Автозапуск SearXNG отключён (GUI_SEARXNG_AUTOSTART=0)")
        return False
    import subprocess

    for cmd in (["docker", "start", "searxng-core"],):
        try:
            done = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        except Exception as exc:
            _log.warning("Не удалось запустить SearXNG: {}", exc)
            return False
        if done.returncode != 0:
            _log.warning("SearXNG не поднят: {}", (done.stderr or "").strip()[:200])
            return False

    for _ in range(20):                       # контейнеру нужно несколько секунд на старт
        if searxng_alive(url, timeout=2):
            _log.info("SearXNG поднят: {}", url)
            return True
        time.sleep(1)
    _log.warning("SearXNG запущен, но не отвечает на {}", url)
    return False


def lightpanda_ready() -> bool:
    """web-scraping ходит по страницам через браузер Lightpanda."""
    import shutil

    return shutil.which("lightpanda") is not None
