from __future__ import annotations

import asyncio
import contextvars
import json
import math
import os
import time
from collections.abc import Mapping
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, cast

from google.adk.models.lite_llm import LiteLlm
from litellm import ModelResponse, ModelResponseStream
from openai import APITimeoutError, AsyncOpenAI
from pydantic import BaseModel

from fedotmas.common.logging import get_logger

if TYPE_CHECKING:
    from google.adk.models.base_llm import BaseLlm

    from fedotmas._settings import ModelConfig

__all__ = ["make_llm"]

_log = get_logger("fedotmas.llm")
_ERROR_PAYLOAD_LEN = 2000
_MAX_TOOL_ARGUMENT_RETRIES = 2
_DEFAULT_REQUEST_TIMEOUT_S = 120.0
_TIMEOUT_RETRY_BACKOFF_S = 0.25
_REQUEST_DEADLINE: contextvars.ContextVar[float | None] = contextvars.ContextVar(
    "fedotmas_llm_request_deadline", default=None
)
_INVALID_TOOL_ARGUMENTS_RETRY = (
    "The previous tool-call arguments were invalid JSON. Return valid, concise "
    "JSON tool arguments."
)


@contextmanager
def llm_request_deadline(deadline: float | None):
    """Expose the current task's monotonic deadline to provider clients."""
    token = _REQUEST_DEADLINE.set(deadline)
    try:
        yield
    finally:
        _REQUEST_DEADLINE.reset(token)


def _request_timeout_from_env() -> float:
    value = os.getenv("FEDOTMAS_LLM_REQUEST_TIMEOUT_S")
    if value is None:
        return _DEFAULT_REQUEST_TIMEOUT_S
    try:
        timeout = float(value)
    except ValueError:
        timeout = 0.0
    if timeout <= 0 or not math.isfinite(timeout):
        _log.warning(
            "Invalid FEDOTMAS_LLM_REQUEST_TIMEOUT_S={!r}; using {}s",
            value,
            _DEFAULT_REQUEST_TIMEOUT_S,
        )
        return _DEFAULT_REQUEST_TIMEOUT_S
    return timeout


def _task_safe_window(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    return max(0.0, remaining - min(1.0, max(0.001, remaining * 0.01)))


class LLMRequestTimeout(RuntimeError):
    """A provider request exceeded its bounded wall-clock window."""

    code = "LLM_REQUEST_TIMEOUT"

    def __init__(self, model: str, timeout: float, elapsed: float):
        super().__init__(
            f"LLM_REQUEST_TIMEOUT: model={model} timeout={timeout:.3f}s "
            f"elapsed={elapsed:.3f}s"
        )
        self.model = model
        self.timeout_seconds = timeout
        self.elapsed_seconds = elapsed


def _json_value(value: Any) -> Any:
    """Convert ADK/Pydantic request objects to JSON-compatible values."""
    if isinstance(value, BaseModel):
        return _json_value(value.model_dump(by_alias=True, exclude_none=True))
    if isinstance(value, Mapping):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_value(item) for item in value]
    return value


def _tool_names_for_log(tools: Any) -> str:
    """Summarize worker-tool registration without serializing full schemas."""
    if tools is None:
        return "none"
    if not tools:
        return "empty"

    names = []
    for tool in tools:
        function = tool.get("function", {}) if isinstance(tool, Mapping) else {}
        name = function.get("name") if isinstance(function, Mapping) else None
        names.append(str(name) if name else "<unnamed>")
    return ",".join(names)


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


def _error_payload(response: Any) -> str:
    try:
        payload = json.dumps(_json_value(response), default=str)
    except Exception:  # noqa: BLE001 - payload diagnostics must survive serialization errors
        return "<unserializable provider response>"
    if len(payload) > _ERROR_PAYLOAD_LEN:
        return payload[:_ERROR_PAYLOAD_LEN] + "... (truncated)"
    return payload


def _invalid_tool_argument_names(response: Any) -> list[str]:
    """Return tool names whose OpenAI-compatible arguments are not JSON."""
    payload = (
        response.model_dump()
        if hasattr(response, "model_dump")
        else _json_value(response)
    )
    choices = payload.get("choices") if isinstance(payload, Mapping) else None
    choices = [] if choices is None else choices
    invalid = []
    for choice in choices:
        message = choice.get("message") if isinstance(choice, Mapping) else None
        message = {} if message is None else message
        calls = message.get("tool_calls") if isinstance(message, Mapping) else None
        calls = [] if calls is None else calls
        for call in calls:
            function = call.get("function", {}) if isinstance(call, Mapping) else {}
            arguments = (
                function.get("arguments") if isinstance(function, Mapping) else None
            )
            try:
                if not isinstance(arguments, str):
                    raise TypeError("arguments must be a JSON string")
                json.loads(arguments)
            except (TypeError, json.JSONDecodeError):
                invalid.append(str(function.get("name", "<unnamed>")))
    return invalid


def _response_shape(response: Any) -> tuple[Any, int, int]:
    """Return finish reason, tool-call count, and text-content length for logging."""
    payload = (
        response.model_dump()
        if hasattr(response, "model_dump")
        else _json_value(response)
    )
    choices = payload.get("choices") if isinstance(payload, Mapping) else None
    choice = choices[0] if choices else {}
    message = choice.get("message") if isinstance(choice, Mapping) else None
    message = {} if message is None else message
    calls = message.get("tool_calls") if isinstance(message, Mapping) else None
    content = message.get("content") if isinstance(message, Mapping) else None
    return (
        choice.get("finish_reason") if isinstance(choice, Mapping) else None,
        len(calls) if isinstance(calls, list) else 0,
        len(content) if isinstance(content, str) else 0,
    )


def _response_usage(response: Any) -> tuple[int, int]:
    payload = (
        response.model_dump()
        if hasattr(response, "model_dump")
        else _json_value(response)
    )
    usage = payload.get("usage") if isinstance(payload, Mapping) else None
    if not isinstance(usage, Mapping):
        return 0, 0
    return (
        int(usage.get("prompt_tokens") or 0),
        int(usage.get("completion_tokens") or 0),
    )


def _attach_usage(error: Exception, prompt_tokens: int, completion_tokens: int) -> None:
    error_with_usage = cast(Any, error)
    error_with_usage.prompt_tokens = prompt_tokens
    error_with_usage.completion_tokens = completion_tokens


class _StreamAdapter:
    """Wraps AsyncOpenAI async stream to yield ``ModelResponseStream`` objects."""

    def __init__(self, stream, *, model: str, timeout: float, deadline: float):
        self._stream = stream
        self._model = model
        self._timeout = timeout
        self._deadline = deadline
        self._started = time.monotonic()
        self._prompt_tokens = 0
        self._completion_tokens = 0

    def __aiter__(self):
        return self

    async def __anext__(self) -> ModelResponseStream:
        remaining = self._deadline - asyncio.get_running_loop().time()
        try:
            if remaining <= 0:
                raise TimeoutError
            chunk = await asyncio.wait_for(self._stream.__anext__(), timeout=remaining)
        except (TimeoutError, APITimeoutError) as exc:
            elapsed = time.monotonic() - self._started
            _log.error(
                "OpenAI-compatible request timed out | model={} timeout={}s elapsed={}s",
                self._model,
                self._timeout,
                round(elapsed, 3),
            )
            error = LLMRequestTimeout(self._model, self._timeout, elapsed)
            _attach_usage(error, self._prompt_tokens, self._completion_tokens)
            raise error from exc
        prompt, completion = _response_usage(chunk)
        self._prompt_tokens += prompt
        self._completion_tokens += completion
        if _finish_reason_is_error(chunk):
            payload = _error_payload(chunk)
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
        self._request_timeout = _request_timeout_from_env()
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=self._request_timeout,
            max_retries=0,
        )
        self._extra_body = dict(extra_body or {})

    def __repr__(self) -> str:
        return f"_ProxyClient(base_url={self._client.base_url!r})"

    async def acompletion(self, model, messages, tools, **kwargs):
        kw = {"model": model, "messages": messages, **kwargs}
        if tools:
            kw["tools"] = tools
        kw.pop("api_base", None)
        kw.pop("api_key", None)
        if self._extra_body:
            kw["extra_body"] = {
                **self._extra_body,
                **kw.get("extra_body", {}),
            }
        stream = kw.get("stream", False)
        task_deadline = _REQUEST_DEADLINE.get()
        started = time.monotonic()
        loop = asyncio.get_running_loop()
        total_timeout = 2 * self._request_timeout + _TIMEOUT_RETRY_BACKOFF_S
        total_deadline = loop.time() + total_timeout
        if task_deadline is not None:
            total_deadline = min(
                total_deadline,
                loop.time() + _task_safe_window(task_deadline),
            )
        request_number = 0
        timeout_retried = False

        async def request(request_kw: dict[str, Any]):
            nonlocal request_number
            request_number += 1
            remaining = total_deadline - loop.time()
            if task_deadline is not None:
                remaining = min(remaining, _task_safe_window(task_deadline))
            timeout = min(self._request_timeout, remaining)
            if timeout <= 0:
                raise LLMRequestTimeout(model, self._request_timeout, time.monotonic() - started)
            request_kw["timeout"] = timeout
            _log.debug(
                "OpenAI-compatible request | model={} request_timeout={}s attempt={} tools={}",
                model,
                round(timeout, 3),
                request_number,
                _tool_names_for_log(tools),
            )
            request_started = time.monotonic()
            try:
                return await asyncio.wait_for(
                    self._client.chat.completions.create(**request_kw), timeout=timeout
                ), timeout
            except (TimeoutError, APITimeoutError) as exc:
                elapsed = time.monotonic() - request_started
                _log.error(
                    "OpenAI-compatible request timed out | model={} timeout={}s elapsed={}s",
                    model,
                    round(timeout, 3),
                    round(elapsed, 3),
                )
                raise LLMRequestTimeout(model, timeout, elapsed) from exc

        async def request_with_timeout_retry(request_kw: dict[str, Any]):
            nonlocal timeout_retried
            try:
                return await request(dict(request_kw))
            except LLMRequestTimeout:
                remaining = total_deadline - loop.time()
                if task_deadline is not None:
                    remaining = min(remaining, _task_safe_window(task_deadline))
                if (
                    stream
                    or timeout_retried
                    or remaining
                    <= _TIMEOUT_RETRY_BACKOFF_S
                    + min(self._request_timeout, 0.001)
                ):
                    raise
                timeout_retried = True
                await asyncio.sleep(_TIMEOUT_RETRY_BACKOFF_S)
                return await request(dict(request_kw))

        resp, active_timeout = await request_with_timeout_retry(kw)
        if stream:
            return _StreamAdapter(
                resp,
                model=model,
                timeout=active_timeout,
                deadline=total_deadline,
            )
        total_prompt_tokens, total_completion_tokens = _response_usage(resp)
        for attempt in range(_MAX_TOOL_ARGUMENT_RETRIES + 1):
            finish_reason, tool_call_count, content_length = _response_shape(resp)
            if isinstance(finish_reason, str) and finish_reason.lower() in {
                "length",
                "max_tokens",
            }:
                _log.warning(
                    "Provider response reached its output-token limit; finish_reason={}",
                    finish_reason,
                )
            _log.debug(
                "OpenAI-compatible response | finish_reason={} tool_calls={} content_chars={}",
                finish_reason,
                tool_call_count,
                content_length,
            )
            invalid = _invalid_tool_argument_names(resp)
            if not invalid:
                break
            if attempt == _MAX_TOOL_ARGUMENT_RETRIES:
                error = RuntimeError(
                    "LLM provider returned malformed JSON tool arguments after "
                    f"{_MAX_TOOL_ARGUMENT_RETRIES + 1} attempts: {', '.join(invalid)}"
                )
                _attach_usage(error, total_prompt_tokens, total_completion_tokens)
                raise error
            _log.warning(
                "Retrying OpenAI-compatible response with invalid JSON tool arguments: {}",
                ", ".join(invalid),
            )
            retry_kw = dict(kw)
            retry_kw["messages"] = [
                *messages,
                {"role": "user", "content": _INVALID_TOOL_ARGUMENTS_RETRY},
            ]
            try:
                resp, active_timeout = await request_with_timeout_retry(retry_kw)
            except Exception as error:
                _attach_usage(error, total_prompt_tokens, total_completion_tokens)
                raise
            prompt_tokens, completion_tokens = _response_usage(resp)
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens
        if _finish_reason_is_error(resp):
            payload = _error_payload(resp)
            _log.error("OpenAI-compatible response finished with error: {}", payload)
            error = RuntimeError(
                f"LLM provider returned finish_reason='error': {payload}"
            )
            _attach_usage(error, total_prompt_tokens, total_completion_tokens)
            raise error
        payload = resp.model_dump()
        if total_prompt_tokens or total_completion_tokens:
            usage = payload.get("usage") or {}
            usage.update(
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                total_tokens=total_prompt_tokens + total_completion_tokens,
            )
            payload["usage"] = usage
        return ModelResponse(**payload)


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
