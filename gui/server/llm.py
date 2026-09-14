"""Прямой клиент провайдера.

Часть работы идёт мимо FEDOT.MAS — ответ одной модели, оценка трудоёмкости,
запасной вызов судьи. Всем им нужен один и тот же клиент.
"""

from __future__ import annotations

import os


def client(model: str) -> tuple:
    """Клиент OpenAI-совместимого API и имя модели.

    Ключ читается из окружения при каждом вызове: в публичном режиме его
    подставляет туда пользователь уже после старта сервера.
    """
    from openai import AsyncOpenAI

    return AsyncOpenAI(base_url=os.getenv("OPENAI_BASE_URL"), api_key=os.getenv("OPENAI_API_KEY")), model
