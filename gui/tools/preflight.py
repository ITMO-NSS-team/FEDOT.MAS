"""Проверка готовности стенда перед живой демонстрацией.

    .venv/bin/python gui/tools/preflight.py            # проверки инфраструктуры (~10 с, бесплатно)
    .venv/bin/python gui/tools/preflight.py --full     # плюс пробный ответ модели и вердикт судьи

Ничего не подкручивает и не подсуживает: только проверяет, что всё, от чего зависит
честный прогон, живо. Каждый пункт — то, что уже ломалось на реальных прогонах.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.parse
import urllib.request

# Порт стенда тот же, что у run.py; GUI_PORT позволяет проверить и публичную копию
BASE = os.getenv("GUI_BASE") or f"http://localhost:{os.getenv('GUI_PORT', '4173')}"
SEARX = os.getenv("SEARXNG_URL", "http://localhost:18888")   # тот же адрес, что у сервера
OPEN_METEO = ("https://archive-api.open-meteo.com/v1/archive?latitude=59.94&longitude=30.31"
              "&start_date=2024-01-01&end_date=2024-01-03&daily=temperature_2m_mean"
              "&timezone=Europe%2FMoscow&format=csv")
# Контрольный запрос, на котором поиск без Яндекса давал ноль результатов
CONTROL_QUERY = "78:06:0220301:6706"

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, note: str = "") -> None:
    RESULTS.append((name, ok, note))
    print(f"  {'✓' if ok else '✗'} {name}" + (f" — {note}" if note else ""))


def get(url: str, timeout: int = 25) -> str:
    return urllib.request.urlopen(url, timeout=timeout).read().decode()


def post(path: str, payload: dict, timeout: int = 300) -> dict:
    req = urllib.request.Request(BASE + path, method="POST",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def main() -> int:
    full = "--full" in sys.argv
    print("Инфраструктура:")

    try:
        st = json.loads(get(f"{BASE}/api/status"))
        check("сервер отвечает", True)
        check("живой режим", bool(st.get("live")))
        check("ключ провайдера задан", bool(st.get("has_key")))
        check("веб-поиск поднят", bool(st.get("web_search")))
        tools = st.get("safe_tools") or []
        need = {"sandbox-light", "download", "document", "websearch-searxng"}
        missing = need - set(tools)
        check("ключевые инструменты на месте", not missing,
              "не хватает: " + ", ".join(sorted(missing)) if missing else f"{len(tools)} шт.")
        check("нерабочий sandbox без ключа E2B исключён", "sandbox" not in tools
              or bool(__import__("os").getenv("E2B_API_KEY")))
    except Exception as exc:
        check("сервер отвечает", False, f"{type(exc).__name__}: {exc}")
        print("\nСервер недоступен — дальше проверять нечего. Запустите gui/run.py.")
        return 1

    try:
        q = urllib.parse.urlencode({"q": CONTROL_QUERY, "format": "json"})
        found = len(json.loads(get(f"{SEARX}/search?{q}", timeout=40)).get("results", []))
        check("поиск находит кадастровый номер", found > 0,
              f"{found} результатов" if found else "0 результатов — включён ли движок yandex?")
    except Exception as exc:
        check("поиск находит кадастровый номер", False, f"{type(exc).__name__}: {exc}")

    try:
        rows = [l for l in get(OPEN_METEO, timeout=40).splitlines() if l.startswith("2024-")]
        check("архив погоды Open-Meteo отдаёт CSV", len(rows) >= 3, f"{len(rows)} строк")
    except Exception as exc:
        check("архив погоды Open-Meteo отдаёт CSV", False, f"{type(exc).__name__}: {exc}")

    if full:
        print("Модели (пробные вызовы, займут минуту-две):")
        try:
            b = post("/api/baseline", {"query": "Сложи 2 и 2, ответь одним числом."})
            check("одиночная модель отвечает", bool(b.get("ok")), b.get("model", ""))
        except Exception as exc:
            check("одиночная модель отвечает", False, str(exc)[:80])
        try:
            j = post("/api/judge", {"query": "Сложи 2+2.", "system_answer": "4",
                                    "single_answer": "5"})
            check("судья выносит вердикт", bool(j.get("ok")), f"победитель: {j.get('winner')}")
            check("судья не ошибся на контрольном случае", j.get("winner") == "system")
        except Exception as exc:
            check("судья выносит вердикт", False, str(exc)[:80])

    failed = [r for r in RESULTS if not r[1]]
    print(f"\nИтог: {len(RESULTS) - len(failed)} из {len(RESULTS)} проверок пройдено"
          + ("" if not failed else " — СТЕНД НЕ ГОТОВ"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
