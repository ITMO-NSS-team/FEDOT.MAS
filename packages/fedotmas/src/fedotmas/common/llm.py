from __future__ import annotations

import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from google.adk.models.lite_llm import LiteLlm
from litellm import ModelResponse, ModelResponseStream
from openai import AsyncOpenAI

from fedotmas.common.logging import get_logger

if TYPE_CHECKING:
    from google.adk.models.base_llm import BaseLlm

    from fedotmas._settings import ModelConfig

__all__ = ["make_llm"]

_log = get_logger("fedotmas.llm")

_OPENAI_SCHEMA_TYPES = {
    "STRING": "string",
    "INTEGER": "integer",
    "NUMBER": "number",
    "BOOLEAN": "boolean",
    "ARRAY": "array",
    "OBJECT": "object",
    "NULL": "null",
}


def _json_value(value: Any) -> Any:
    """Convert ADK/Pydantic request objects to JSON-compatible values."""
    if hasattr(value, "model_dump"):
        return _json_value(value.model_dump(by_alias=True, exclude_none=True))
    if isinstance(value, Mapping):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_value(item) for item in value]
    return value


def _normalize_openai_schema(value: Any) -> Any:
    """Normalize ADK schema conventions accepted by OpenAI-compatible APIs."""
    if isinstance(value, list):
        return [_normalize_openai_schema(item) for item in value]
    if not isinstance(value, dict):
        return value

    normalized = {
        ("additionalProperties" if key == "additional_properties" else key):
        _normalize_openai_schema(item)
        for key, item in value.items()
    }
    schema_type = normalized.get("type")
    if isinstance(schema_type, str):
        normalized["type"] = _OPENAI_SCHEMA_TYPES.get(schema_type.upper(), schema_type)
    return normalized


def _normalize_openai_tools(tools: Any) -> list[dict[str, Any]] | None:
    """Translate ADK-generated function declarations to OpenAI tool payloads.

    ADK versions emit schemas with Google-style uppercase types and may retain
    ``response`` metadata on agent tools. OpenAI-compatible endpoints reject
    those forms even though they accept ordinary function tools.
    """
    if not tools:
        return None

    normalized_tools: list[dict[str, Any]] = []
    for tool in _json_value(tools):
        if not isinstance(tool, dict):
            raise TypeError(f"Unsupported tool declaration type: {type(tool)!r}")
        normalized_tool = _normalize_openai_schema(tool)
        function = normalized_tool.get("function")
        if isinstance(function, dict):
            # These describe an ADK agent-tool return value, not an OpenAI
            # Chat Completions function parameter, and are rejected by many
            # OpenAI-compatible gateways.
            function.pop("response", None)
            function.pop("response_json_schema", None)
            parameters_json_schema = function.pop("parameters_json_schema", None)
            if parameters_json_schema is not None and "parameters" not in function:
                function["parameters"] = parameters_json_schema
        normalized_tools.append(normalized_tool)
    return normalized_tools


def _finish_reason_is_error(response: Any) -> bool:
    choices = getattr(response, "choices", None)
    if choices is None and isinstance(response, Mapping):
        choices = response.get("choices")
    if not choices:
        return False
    choice = choices[0]
    reason = getattr(choice, "finish_reason", None)
    if reason is None and isinstance(choice, Mapping):
        reason = choice.get("finish_reason")
    return isinstance(reason, str) and reason.lower() == "error"


def _response_payload(response: Any) -> Any:
    return _json_value(response)


class _StreamAdapter:
    """Wraps AsyncOpenAI async stream to yield ``ModelResponseStream`` objects."""

    def __init__(self, stream):
        self._stream = stream

    def __aiter__(self):
        return self

    async def __anext__(self) -> ModelResponseStream:
        chunk = await self._stream.__anext__()
        if _finish_reason_is_error(chunk):
            payload = _response_payload(chunk)
            _log.error(
                "OpenAI-compatible streaming response finished with error: {}", payload
            )
            raise RuntimeError(
                f"LLM provider returned finish_reason='error': {payload}"
            )
        return ModelResponseStream(**chunk.model_dump())


class _ProxyClient:
    """Direct OpenAI-compatible transport, bypassing litellm routing.

    Implements the same ``acompletion`` contract as ``LiteLLMClient``
    but sends requests directly via ``AsyncOpenAI`` so model names
    pass through as-is to the proxy.
    """

    def __init__(self, base_url: str, api_key: str, extra_body: dict[str, Any] | None):
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self._extra_body = dict(extra_body or {})

    def __repr__(self) -> str:
        return f"_ProxyClient(base_url={self._client.base_url!r})"

    async def acompletion(self, model, messages, tools, **kwargs):
        kw = {"model": model, "messages": messages, **kwargs}
        normalized_tools = _normalize_openai_tools(tools)
        if normalized_tools:
            kw["tools"] = normalized_tools
        kw.pop("api_base", None)
        kw.pop("api_key", None)
        if self._extra_body:
            kw["extra_body"] = {
                **self._extra_body,
                **kw.get("extra_body", {}),
            }
        stream = kw.get("stream", False)
        _log.debug(
            "OpenAI-compatible request | model={} tools={}",
            model,
            json.dumps(
                normalized_tools, ensure_ascii=False, sort_keys=True, default=str
            ),
        )
        resp = await self._client.chat.completions.create(**kw)
        if _finish_reason_is_error(resp):
            payload = _response_payload(resp)
            _log.error("OpenAI-compatible response finished with error: {}", payload)
            raise RuntimeError(
                f"LLM provider returned finish_reason='error': {payload}"
            )
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
            extra_body=cfg.extra_body,
        )
        return llm

    kwargs: dict[str, Any] = {}
    if cfg.api_key:
        kwargs["api_key"] = cfg.api_key
    if cfg.extra_body:
        kwargs["extra_body"] = cfg.extra_body
    return LiteLlm(model=cfg.model, **kwargs)
