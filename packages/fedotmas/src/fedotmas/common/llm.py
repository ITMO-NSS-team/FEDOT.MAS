from __future__ import annotations

from typing import TYPE_CHECKING, Any

from google.adk.models.lite_llm import LiteLlm
from litellm import ModelResponse, ModelResponseStream
from openai import AsyncOpenAI

if TYPE_CHECKING:
    from google.adk.models.base_llm import BaseLlm

    from fedotmas.config.settings import ModelConfig

__all__ = ["make_llm"]


class _StreamAdapter:
    """Wraps AsyncOpenAI async stream to yield ``ModelResponseStream`` objects."""

    def __init__(self, stream):
        self._stream = stream

    def __aiter__(self):
        return self

    async def __anext__(self) -> ModelResponseStream:
        chunk = await self._stream.__anext__()
        return ModelResponseStream(**chunk.model_dump())


class _ProxyClient:
    """Direct OpenAI-compatible transport, bypassing litellm routing.

    Implements the same ``acompletion`` contract as ``LiteLLMClient``
    but sends requests directly via ``AsyncOpenAI`` so model names
    pass through as-is to the proxy.
    """

    def __init__(self, base_url: str, api_key: str):
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    async def acompletion(self, model, messages, tools, **kwargs):
        kw = {"model": model, "messages": messages, **kwargs}
        if tools:
            kw["tools"] = tools
        kw.pop("api_base", None)
        kw.pop("api_key", None)
        stream = kw.get("stream", False)
        resp = await self._client.chat.completions.create(**kw)
        if stream:
            return _StreamAdapter(resp)
        return ModelResponse(**resp.model_dump())


def make_llm(cfg: ModelConfig) -> BaseLlm:
    """Create a ``LiteLlm`` instance from *cfg*.

    When *cfg.api_base* is set (proxy mode), replaces the default litellm
    transport with ``_ProxyClient`` so model names pass through as-is.
    """
    if cfg.api_base:
        llm = LiteLlm(model=cfg.model)
        llm.llm_client = _ProxyClient(  # type: ignore
            base_url=cfg.api_base,
            api_key=cfg.api_key or "no-key",
        )
        return llm

    kwargs: dict[str, Any] = {}
    if cfg.api_key:
        kwargs["api_key"] = cfg.api_key
    return LiteLlm(model=cfg.model, **kwargs)
