"""Direct GUI model calls use the same selective OpenRouter transport as agents."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import openai
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
gui_llm = importlib.import_module("server.llm")


def test_openrouter_model_uses_proxy_and_unprefixed_name(monkeypatch):
    monkeypatch.setenv("FEDOTMAS_OPENROUTER_PROXY_URL", "http://10.32.11.45:7890")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    proxy_client = MagicMock()
    sdk_client = MagicMock()
    monkeypatch.setattr(httpx, "AsyncClient", MagicMock(return_value=proxy_client))
    sdk_factory = MagicMock(return_value=sdk_client)
    monkeypatch.setattr(openai, "AsyncOpenAI", sdk_factory)

    client, model = gui_llm.client("openrouter/qwen/qwen3")

    assert client is sdk_client
    assert model == "qwen/qwen3"
    sdk_factory.assert_called_once_with(
        base_url="https://openrouter.ai/api/v1",
        api_key="sk-test",
        http_client=proxy_client,
    )


def test_non_openrouter_backend_stays_direct(monkeypatch):
    monkeypatch.setenv("FEDOTMAS_OPENROUTER_PROXY_URL", "http://10.32.11.45:7890")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:9090/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    proxy_factory = MagicMock()
    sdk_factory = MagicMock()
    monkeypatch.setattr(httpx, "AsyncClient", proxy_factory)
    monkeypatch.setattr(openai, "AsyncOpenAI", sdk_factory)

    _, model = gui_llm.client("openai/gpt-4o")

    assert model == "openai/gpt-4o"
    proxy_factory.assert_not_called()
    sdk_factory.assert_called_once_with(
        base_url="http://localhost:9090/v1", api_key="sk-test"
    )


def test_openrouter_base_url_uses_proxy_for_unprefixed_gui_model(monkeypatch):
    monkeypatch.setenv("FEDOTMAS_OPENROUTER_PROXY_URL", "http://10.32.11.45:7890")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    proxy_client = MagicMock()
    sdk_factory = MagicMock()
    monkeypatch.setattr(httpx, "AsyncClient", MagicMock(return_value=proxy_client))
    monkeypatch.setattr(openai, "AsyncOpenAI", sdk_factory)

    _, model = gui_llm.client("openai/gpt-4o")

    assert model == "openai/gpt-4o"
    sdk_factory.assert_called_once_with(
        base_url="https://openrouter.ai/api/v1",
        api_key="sk-test",
        http_client=proxy_client,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("requested,expected", [(None, 4096), (16000, 16000)])
async def test_direct_call_always_limits_completion(monkeypatch, requested, expected):
    sdk_client = MagicMock()
    sdk_client.__aenter__ = AsyncMock(return_value=sdk_client)
    sdk_client.__aexit__ = AsyncMock(return_value=None)
    sdk_client.chat.completions.create = AsyncMock(return_value=SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=2, completion_tokens=1),
        choices=[SimpleNamespace(message=SimpleNamespace(content="готово"))],
    ))
    monkeypatch.setattr(gui_llm, "client", lambda model: (sdk_client, "qwen/qwen3"))

    response = await gui_llm.complete(
        "openrouter/qwen/qwen3", [{"role": "user", "content": "Тест"}],
        max_tokens=requested,
    )

    assert response.text == "готово"
    assert sdk_client.chat.completions.create.await_args.kwargs["max_tokens"] == expected
