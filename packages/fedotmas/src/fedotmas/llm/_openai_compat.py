from __future__ import annotations

import json
from abc import abstractmethod
from typing import Any, AsyncGenerator

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from openai import AsyncOpenAI
from pydantic import BaseModel, PrivateAttr

from fedotmas.common.logging import get_logger

_log = get_logger("fedotmas.llm._openai_compat")

_FINISH_REASON_MAP: dict[str, types.FinishReason] = {
    "stop": types.FinishReason.STOP,
    "length": types.FinishReason.MAX_TOKENS,
    "tool_calls": types.FinishReason.STOP,
    "function_call": types.FinishReason.STOP,
    "content_filter": types.FinishReason.SAFETY,
}

_MISSING_TOOL_RESULT = (
    "Error: Missing tool result (tool execution may have been interrupted "
    "before a response was recorded)."
)


def _safe_json(obj: Any) -> str:
    """Serialize *obj* to a JSON string; fall back to ``str()``."""
    try:
        return json.dumps(obj, default=str)
    except (TypeError, ValueError):
        _log.warning("Failed to serialize to JSON, falling back to str(): {}", type(obj))
        return str(obj)


def _map_finish_reason(reason: str | None) -> types.FinishReason:
    if not reason:
        _log.debug("No finish_reason in response, assuming STOP")
        return types.FinishReason.STOP
    return _FINISH_REASON_MAP.get(reason.lower(), types.FinishReason.OTHER)


def _schema_to_dict(schema: types.Schema | dict[str, Any]) -> dict[str, Any]:
    """Recursively convert a ``types.Schema`` (or dict) to a plain dict."""
    schema_dict: dict[str, Any] = (
        schema.model_dump(exclude_none=True)
        if isinstance(schema, types.Schema)
        else dict(schema)
    )

    enum_values = schema_dict.get("enum")
    if isinstance(enum_values, (list, tuple)):
        schema_dict["enum"] = [v for v in enum_values if v is not None]

    if "type" in schema_dict and schema_dict["type"] is not None:
        t = schema_dict["type"]
        raw = t.value if isinstance(t, types.Type) else str(t)
        schema_dict["type"] = raw.lower()

    if "items" in schema_dict:
        items = schema_dict["items"]
        if isinstance(items, (types.Schema, dict)):
            schema_dict["items"] = _schema_to_dict(items)

    if "properties" in schema_dict:
        schema_dict["properties"] = {
            k: _schema_to_dict(v) if isinstance(v, (types.Schema, dict)) else v
            for k, v in schema_dict["properties"].items()
        }

    return schema_dict


def _content_to_messages(content: types.Content) -> list[dict[str, Any]]:
    """Convert a single ``types.Content`` to one or more OpenAI message dicts."""
    tool_msgs: list[dict[str, Any]] = []
    non_tool_parts: list[types.Part] = []

    for part in content.parts or []:
        if part.function_response:
            resp = part.function_response.response
            tool_msgs.append({
                "role": "tool",
                "tool_call_id": part.function_response.id,
                "content": resp if isinstance(resp, str) else _safe_json(resp),
            })
        else:
            non_tool_parts.append(part)

    if tool_msgs and not non_tool_parts:
        return tool_msgs

    role = "user" if content.role == "user" else "assistant"

    if role == "user":
        texts = [p.text for p in non_tool_parts if p.text and not p.thought]
        msg: dict[str, Any] = {"role": "user", "content": "\n".join(texts) or ""}
        result = tool_msgs + [msg] if tool_msgs else [msg]
        return result

    tool_calls: list[dict[str, Any]] = []
    text_parts: list[str] = []
    for part in non_tool_parts:
        if part.function_call:
            tool_calls.append({
                "type": "function",
                "id": part.function_call.id,
                "function": {
                    "name": part.function_call.name,
                    "arguments": _safe_json(part.function_call.args),
                },
            })
        elif part.text and not part.thought:
            text_parts.append(part.text)

    msg = {"role": "assistant", "content": "\n".join(text_parts) or None}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    result = tool_msgs + [msg] if tool_msgs else [msg]
    return result


def _adk_contents_to_messages(
    contents: list[types.Content],
    system_instruction: str | None = None,
) -> list[dict[str, Any]]:
    """Convert ADK content list + system instruction to OpenAI messages."""
    messages: list[dict[str, Any]] = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    for content in contents:
        messages.extend(_content_to_messages(content))
    return messages


def _adk_tools_to_openai(
    tools: Any,
) -> list[dict[str, Any]] | None:
    """Convert ADK tool declarations to OpenAI tool format."""
    if not tools:
        return None
    result: list[dict[str, Any]] = []
    for tool in tools:
        if not tool.function_declarations:
            continue
        for fd in tool.function_declarations:
            assert fd.name
            parameters: dict[str, Any] = {"type": "object", "properties": {}}
            if fd.parameters and fd.parameters.properties:
                properties = {
                    k: _schema_to_dict(v)
                    for k, v in fd.parameters.properties.items()
                }
                parameters = {"type": "object", "properties": properties}
                required = getattr(fd.parameters, "required", None)
                if required:
                    parameters["required"] = required
            elif fd.parameters_json_schema:
                parameters = fd.parameters_json_schema

            result.append({
                "type": "function",
                "function": {
                    "name": fd.name,
                    "description": fd.description or "",
                    "parameters": parameters,
                },
            })
    return result or None


def _response_schema_to_format(
    response_schema: Any,
) -> dict[str, Any] | None:
    """Convert an ADK response schema to OpenAI ``response_format``."""
    if response_schema is None:
        return None

    schema_name = "response"

    if isinstance(response_schema, dict):
        structured_types = {"json_object", "json_schema"}
        schema_type = response_schema.get("type")
        if isinstance(schema_type, str) and schema_type.lower() in structured_types:
            return response_schema
        schema_dict = dict(response_schema)
        if "title" in schema_dict:
            schema_name = str(schema_dict["title"])
    elif isinstance(response_schema, type) and issubclass(response_schema, BaseModel):
        schema_dict = response_schema.model_json_schema()
        schema_name = response_schema.__name__
    elif isinstance(response_schema, BaseModel):
        if isinstance(response_schema, types.Schema):
            schema_dict = response_schema.model_dump(exclude_none=True, mode="json")
            if "title" in schema_dict:
                schema_name = str(schema_dict["title"])
        else:
            schema_dict = response_schema.__class__.model_json_schema()
            schema_name = response_schema.__class__.__name__
    elif hasattr(response_schema, "model_dump"):
        schema_dict = response_schema.model_dump(exclude_none=True, mode="json")
        schema_name = response_schema.__class__.__name__
    else:
        _log.warning(
            "Unsupported response_schema type: {}", type(response_schema)
        )
        return None

    if (
        isinstance(schema_dict, dict)
        and schema_dict.get("type") == "object"
        and "additionalProperties" not in schema_dict
    ):
        schema_dict = dict(schema_dict)
        schema_dict["additionalProperties"] = False

    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema_name,
            "strict": True,
            "schema": schema_dict,
        },
    }


def _config_to_params(config: types.GenerateContentConfig) -> dict[str, Any]:
    """Extract generation params from ADK config, renaming as needed."""
    params: dict[str, Any] = {}
    config_dict = config.model_dump(exclude_none=True)
    rename = {
        "max_output_tokens": "max_completion_tokens",
        "stop_sequences": "stop",
    }
    for key in (
        "temperature",
        "max_output_tokens",
        "top_p",
        "top_k",
        "stop_sequences",
        "presence_penalty",
        "frequency_penalty",
    ):
        if key in config_dict:
            params[rename.get(key, key)] = config_dict[key]
    return params


def _ensure_tool_results(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Insert placeholder tool messages for any missing tool results."""
    if not messages:
        return messages

    healed: list[dict[str, Any]] = []
    pending_ids: list[str] = []

    for msg in messages:
        role = msg.get("role")

        if pending_ids and role != "tool":
            _log.warning("Missing tool results for tool_call_id(s): {}", pending_ids)
            healed.extend(
                {"role": "tool", "tool_call_id": tid, "content": _MISSING_TOOL_RESULT}
                for tid in pending_ids
            )
            pending_ids = []

        if role == "assistant":
            tool_calls = msg.get("tool_calls") or []
            pending_ids = [
                tc.get("id") for tc in tool_calls if tc.get("id")
            ]
        elif role == "tool":
            tid = msg.get("tool_call_id")
            if tid in pending_ids:
                pending_ids.remove(tid)

        healed.append(msg)

    if pending_ids:
        _log.warning("Missing tool results for tool_call_id(s): {}", pending_ids)
        healed.extend(
            {"role": "tool", "tool_call_id": tid, "content": _MISSING_TOOL_RESULT}
            for tid in pending_ids
        )

    return healed


def _openai_response_to_llm_response(
    response: Any,
) -> LlmResponse:
    """Convert an ``openai.ChatCompletion`` to an ADK ``LlmResponse``."""
    choice = response.choices[0] if response.choices else None
    if not choice:
        raise ValueError("No choices in OpenAI response")

    message = choice.message
    parts: list[types.Part] = []

    if message.content:
        parts.append(types.Part.from_text(text=message.content))

    if message.tool_calls:
        for tc in message.tool_calls:
            if tc.type == "function":
                part = types.Part.from_function_call(
                    name=tc.function.name,
                    args=json.loads(tc.function.arguments or "{}"),
                )
                if part.function_call:
                    part.function_call.id = tc.id
                parts.append(part)

    finish_reason = _map_finish_reason(
        choice.finish_reason if choice.finish_reason else None
    )

    usage_metadata = None
    if response.usage:
        usage_metadata = types.GenerateContentResponseUsageMetadata(
            prompt_token_count=response.usage.prompt_tokens or 0,
            candidates_token_count=response.usage.completion_tokens or 0,
            total_token_count=response.usage.total_tokens or 0,
        )

    return LlmResponse(
        content=types.Content(role="model", parts=parts) if parts else None,
        partial=False,
        model_version=response.model,
        finish_reason=finish_reason,
        usage_metadata=usage_metadata,
    )


class OpenAICompatibleLlm(BaseLlm):
    """Base ``BaseLlm`` for any OpenAI-compatible endpoint.

    Subclasses must implement ``_build_client`` and optionally override
    ``_extra_create_kwargs`` and ``_resolve_model``.
    """

    api_key: str | None = None
    api_base: str | None = None

    _client: AsyncOpenAI | None = PrivateAttr(default=None)

    @abstractmethod
    def _build_client(self) -> AsyncOpenAI:
        """Create the ``AsyncOpenAI`` client."""

    def _extra_create_kwargs(self) -> dict[str, Any]:
        """Additional kwargs for ``client.chat.completions.create``."""
        return {}

    def _resolve_model(self, raw_model: str) -> str:
        """Strip provider prefix or transform model name."""
        return raw_model

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = self._build_client()
        return self._client

    async def generate_content_async(
        self,
        llm_request: LlmRequest,
        stream: bool = False,
    ) -> AsyncGenerator[LlmResponse, None]:
        self._maybe_append_user_content(llm_request)

        effective_model = self._resolve_model(llm_request.model or self.model)

        raw_sys = (
            llm_request.config.system_instruction
            if llm_request.config
            else None
        )
        if isinstance(raw_sys, str):
            system_instruction: str | None = raw_sys
        elif raw_sys is not None:
            sys_parts = getattr(raw_sys, "parts", None) or []
            system_instruction = "\n".join(
                p.text for p in sys_parts if p.text
            ) or None
        else:
            system_instruction = None

        messages = _adk_contents_to_messages(
            llm_request.contents, system_instruction
        )
        messages = _ensure_tool_results(messages)

        tools = None
        if llm_request.config and llm_request.config.tools:
            raw_tools = llm_request.config.tools
            tools = _adk_tools_to_openai(raw_tools)

        response_format = None
        if llm_request.config and llm_request.config.response_schema:
            response_format = _response_schema_to_format(
                llm_request.config.response_schema
            )

        create_kwargs: dict[str, Any] = {
            "model": effective_model,
            "messages": messages,
        }
        if tools:
            create_kwargs["tools"] = tools
        if response_format:
            create_kwargs["response_format"] = response_format

        if llm_request.config:
            create_kwargs.update(_config_to_params(llm_request.config))

        create_kwargs.update(self._extra_create_kwargs())

        client = self._get_client()

        if stream:
            async for resp in self._stream(client, create_kwargs):
                yield resp
        else:
            response = await client.chat.completions.create(**create_kwargs)
            yield _openai_response_to_llm_response(response)

    async def _stream(
        self,
        client: AsyncOpenAI,
        create_kwargs: dict[str, Any],
    ) -> AsyncGenerator[LlmResponse, None]:
        create_kwargs["stream"] = True
        create_kwargs["stream_options"] = {"include_usage": True}

        text = ""
        function_calls: dict[int, dict[str, Any]] = {}
        usage_metadata = None
        # fallback_index tracks tool call index when tc_delta.index is None
        fallback_index = 0
        model_version: str | None = None

        async for chunk in await client.chat.completions.create(**create_kwargs):
            model_version = chunk.model or model_version
            if not chunk.choices:
                if chunk.usage:
                    usage_metadata = types.GenerateContentResponseUsageMetadata(
                        prompt_token_count=chunk.usage.prompt_tokens or 0,
                        candidates_token_count=chunk.usage.completion_tokens or 0,
                        total_token_count=chunk.usage.total_tokens or 0,
                    )
                continue

            choice = chunk.choices[0]
            delta = choice.delta

            if delta and delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index if tc_delta.index is not None else fallback_index
                    if idx not in function_calls:
                        function_calls[idx] = {"name": "", "args": "", "id": None}
                    if tc_delta.function and tc_delta.function.name:
                        function_calls[idx]["name"] += tc_delta.function.name
                    if tc_delta.function and tc_delta.function.arguments:
                        function_calls[idx]["args"] += tc_delta.function.arguments
                        try:
                            json.loads(function_calls[idx]["args"])
                            fallback_index += 1
                        except json.JSONDecodeError:
                            pass
                    if tc_delta.id:
                        function_calls[idx]["id"] = tc_delta.id

            if delta and delta.content:
                text += delta.content
                yield LlmResponse(
                    content=types.Content(
                        role="model",
                        parts=[types.Part.from_text(text=delta.content)],
                    ),
                    partial=True,
                    model_version=model_version,
                )

            if chunk.usage:
                usage_metadata = types.GenerateContentResponseUsageMetadata(
                    prompt_token_count=chunk.usage.prompt_tokens or 0,
                    candidates_token_count=chunk.usage.completion_tokens or 0,
                    total_token_count=chunk.usage.total_tokens or 0,
                )

        parts: list[types.Part] = []
        if text:
            parts.append(types.Part.from_text(text=text))
        for _, fc in sorted(function_calls.items()):
            if fc["id"]:
                part = types.Part.from_function_call(
                    name=fc["name"],
                    args=json.loads(fc["args"] or "{}"),
                )
                if part.function_call:
                    part.function_call.id = fc["id"]
                parts.append(part)

        final = LlmResponse(
            content=types.Content(role="model", parts=parts) if parts else None,
            partial=False,
            model_version=model_version,
            finish_reason=types.FinishReason.STOP,
            usage_metadata=usage_metadata,
        )
        yield final
