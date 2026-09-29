"""Запуск стенда.

    python gui/run.py                 # локально, модели host/* берутся из Codex CLI
    GUI_PUBLIC=1 python gui/run.py    # наружу: ключ вводит пользователь, нужен токен

Порт задаётся переменной GUI_PORT (по умолчанию 4173). Слушаем только 127.0.0.1:
наружу стенд отдаётся туннелем, а не открытым портом — так его нельзя случайно
выставить в локальную сеть.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import uvicorn
from server.app import app
from server.config import ACCESS_TOKEN, PUBLIC_MODE


def main() -> None:
    port = int(os.getenv("GUI_PORT", "4173"))
    host = os.getenv("GUI_HOST", "127.0.0.1")
    if PUBLIC_MODE:
        # Токен печатаем один раз при старте: он и есть ключ от адреса.
        print(f"FEDOT.MAS GUI (публичный режим): http://localhost:{port}/?t={ACCESS_TOKEN}")
        print("Ключ провайдера сервер не хранит — его вводит пользователь в интерфейсе.")
    else:
        print(f"FEDOT.MAS GUI (живой режим, Codex subscription): http://localhost:{port}")
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
