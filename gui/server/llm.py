"""Прямые одноагентные вызовы для GUI.

``host/...`` выполняется через авторизованный Codex CLI. Остальные имена
сохраняют прежний OpenAI-совместимый transport.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from fedotmas.common.codex_cli import is_codex_model, run_codex_cli


@dataclass(frozen=True)
class Completion:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


def client(model: str) -> tuple:
    """Клиент OpenAI-совместимого API и имя модели.

    Ключ читается из окружения при каждом вызове: в публичном режиме его
    подставляет туда пользователь уже после старта сервера. Модели
    ``openrouter/...`` ходят на OpenRouter напрямую — префикс убираем,
    остаток и есть имя модели у провайдера.
    """
    from openai import AsyncOpenAI

    if model.startswith("openrouter/"):
        return AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY"),
        ), model.removeprefix("openrouter/")

    return AsyncOpenAI(base_url=os.getenv("OPENAI_BASE_URL"), api_key=os.getenv("OPENAI_API_KEY")), model


async def complete(
    model: str,
    messages: list[dict[str, str]],
    *,
    json_schema: dict[str, Any] | None = None,
    max_tokens: int | None = None,
) -> Completion:
    """Return one complete response through the selected model transport."""
    if is_codex_model(model):
        prompt = "\n\n".join(
            f"[{item.get('role', 'user')}]\n{item.get('content', '')}"
            for item in messages
        )
        prompt = (
            "You are answering one GUI model request. Follow the messages below. "
            "Do not inspect files and do not use Codex shell tools. Return only the "
            "requested answer.\n\n" + prompt
        )
        result = await run_codex_cli(model, prompt, output_schema=json_schema)
        return Completion(
            text=result.text,
            model=model,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
        )

    api_client, resolved = client(model)
    kwargs: dict[str, Any] = {"model": resolved, "messages": messages}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if json_schema is not None:
        kwargs["response_format"] = {"type": "json_object"}
    response = await api_client.chat.completions.create(**kwargs)
    usage = response.usage
    return Completion(
        text=response.choices[0].message.content or "",
        model=resolved,
        prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
    )
