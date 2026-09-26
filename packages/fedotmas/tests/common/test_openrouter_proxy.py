"""OpenRouter's dedicated proxy must never intercept unrelated API calls."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
from fedotmas._settings import ModelConfig
from fedotmas.common.llm import _ProxyClient, make_llm
from fedotmas.common.openrouter_proxy import (
    is_openrouter_url,
    openrouter_http_client,
)


def test_proxy_disabled_by_default(monkeypatch):
    monkeypatch.delenv("FEDOTMAS_OPENROUTER_PROXY_URL", raising=False)
    assert openrouter_http_client("https://openrouter.ai/api/v1") is None


def test_proxy_uses_only_openrouter_https(monkeypatch):
    monkeypatch.setenv("FEDOTMAS_OPENROUTER_PROXY_URL", "http://10.32.11.45:7890")
    mock_client = MagicMock()
    monkeypatch.setattr(httpx, "AsyncClient", mock_client)

    assert is_openrouter_url("https://openrouter.ai/api/v1")
    for url in (
        None,
        "http://openrouter.ai/api/v1",
        "https://openrouter.ai.evil.example/api/v1",
        "https://api.openai.com/v1",
        "http://localhost:9090/v1",
    ):
        assert openrouter_http_client(url) is None
    mock_client.assert_not_called()

    assert (
        openrouter_http_client("https://openrouter.ai/api/v1")
        is mock_client.return_value
    )
    mock_client.assert_called_once_with(
        proxy="http://10.32.11.45:7890", follow_redirects=True
    )


def test_core_uses_proxy_for_openrouter_without_base_url(monkeypatch):
    monkeypatch.setenv("FEDOTMAS_OPENROUTER_PROXY_URL", "http://10.32.11.45:7890")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    llm = make_llm(ModelConfig(model="openrouter/openai/gpt-4o"))
    assert isinstance(llm.llm_client, _ProxyClient)
    assert str(llm.llm_client._client.base_url) == "https://openrouter.ai/api/v1/"
    assert llm.llm_client._client.api_key == "sk-test"


def test_explicit_openrouter_url_uses_openrouter_key(monkeypatch):
    monkeypatch.setenv("FEDOTMAS_OPENROUTER_PROXY_URL", "http://10.32.11.45:7890")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    llm = make_llm(
        ModelConfig(
            model="openrouter/openai/gpt-4o", api_base="https://openrouter.ai/api/v1"
        )
    )
    assert llm.llm_client._client.api_key == "sk-test"


async def test_provider_prefix_removed_only_for_openrouter():
    client = _ProxyClient("https://openrouter.ai/api/v1", "test", None)
    client._client.chat.completions.create = AsyncMock(return_value=MagicMock())
    await client.acompletion("openrouter/openai/gpt-4o", [], [], stream=True)
    assert (
        client._client.chat.completions.create.await_args.kwargs["model"]
        == "openai/gpt-4o"
    )
    await client._client.close()


async def test_provider_prefix_kept_for_other_compatible_apis():
    client = _ProxyClient("http://localhost:9090/v1", "test", None)
    client._client.chat.completions.create = AsyncMock(return_value=MagicMock())
    await client.acompletion("openrouter/openai/gpt-4o", [], [], stream=True)
    assert (
        client._client.chat.completions.create.await_args.kwargs["model"]
        == "openrouter/openai/gpt-4o"
    )
    await client._client.close()
