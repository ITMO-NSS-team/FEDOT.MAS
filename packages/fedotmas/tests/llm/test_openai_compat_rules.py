"""Conversion function unit tests — pure functions, no mocking, no network."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from google.genai import types

from fedotmas.llm._openai_compat import (
    _adk_contents_to_messages,
    _adk_tools_to_openai,
    _config_to_params,
    _content_to_messages,
    _ensure_tool_results,
    _map_finish_reason,
    _openai_response_to_llm_response,
    _response_schema_to_format,
    _schema_to_dict,
)


# ---- System instruction insertion -----------------------------------------


class TestSystemInstructionInsertion:
    """System prompt → first message with role=system."""

    def test_system_instruction_prepended(self):
        msgs = _adk_contents_to_messages([], system_instruction="You are helpful.")
        assert msgs[0] == {"role": "system", "content": "You are helpful."}

    def test_no_system_instruction(self):
        msgs = _adk_contents_to_messages([])
        assert msgs == []

    def test_system_before_user(self):
        user = types.Content(
            role="user",
            parts=[types.Part.from_text(text="hi")],
        )
        msgs = _adk_contents_to_messages([user], system_instruction="sys")
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"


# ---- User content conversion ----------------------------------------------


class TestUserContentConversion:
    """Text parts → user message."""

    def test_single_text(self):
        content = types.Content(
            role="user",
            parts=[types.Part.from_text(text="hello")],
        )
        msgs = _content_to_messages(content)
        assert len(msgs) == 1
        assert msgs[0] == {"role": "user", "content": "hello"}

    def test_multiple_text_parts(self):
        content = types.Content(
            role="user",
            parts=[
                types.Part.from_text(text="line 1"),
                types.Part.from_text(text="line 2"),
            ],
        )
        msgs = _content_to_messages(content)
        assert msgs[0]["content"] == "line 1\nline 2"


# ---- Assistant + tool_calls conversion ------------------------------------


class TestAssistantToolCallConversion:
    """function_call parts → assistant message with tool_calls."""

    def test_function_call(self):
        part = types.Part.from_function_call(
            name="get_weather",
            args={"city": "SF"},
        )
        part.function_call.id = "call_123"
        content = types.Content(role="model", parts=[part])
        msgs = _content_to_messages(content)
        assert len(msgs) == 1
        msg = msgs[0]
        assert msg["role"] == "assistant"
        assert len(msg["tool_calls"]) == 1
        tc = msg["tool_calls"][0]
        assert tc["type"] == "function"
        assert tc["id"] == "call_123"
        assert tc["function"]["name"] == "get_weather"
        assert json.loads(tc["function"]["arguments"]) == {"city": "SF"}

    def test_text_and_function_call(self):
        parts = [
            types.Part.from_text(text="Let me check."),
            types.Part.from_function_call(name="lookup", args={"q": "x"}),
        ]
        parts[1].function_call.id = "call_456"
        content = types.Content(role="model", parts=parts)
        msgs = _content_to_messages(content)
        assert len(msgs) == 1
        msg = msgs[0]
        assert msg["content"] == "Let me check."
        assert len(msg["tool_calls"]) == 1


# ---- Function response conversion -----------------------------------------


class TestFunctionResponseConversion:
    """function_response → tool role messages."""

    def test_single_function_response(self):
        part = types.Part.from_function_response(
            name="get_weather",
            response={"temp": 72},
        )
        part.function_response.id = "call_123"
        content = types.Content(role="user", parts=[part])
        msgs = _content_to_messages(content)
        assert len(msgs) == 1
        assert msgs[0]["role"] == "tool"
        assert msgs[0]["tool_call_id"] == "call_123"
        assert json.loads(msgs[0]["content"]) == {"temp": 72}

    def test_dict_response(self):
        part = types.Part.from_function_response(
            name="echo",
            response={"text": "hello"},
        )
        part.function_response.id = "call_789"
        content = types.Content(role="user", parts=[part])
        msgs = _content_to_messages(content)
        parsed = json.loads(msgs[0]["content"])
        assert parsed == {"text": "hello"}


# ---- Tool declaration conversion ------------------------------------------


class TestToolDeclarationConversion:
    """FunctionDeclaration → OpenAI tool format."""

    def test_basic_tool(self):
        tool = types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name="search",
                    description="Search the web",
                    parameters=types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            "query": types.Schema(type=types.Type.STRING),
                        },
                    ),
                )
            ]
        )
        result = _adk_tools_to_openai([tool])
        assert result is not None
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "search"
        assert "query" in result[0]["function"]["parameters"]["properties"]

    def test_no_tools_returns_none(self):
        assert _adk_tools_to_openai(None) is None
        assert _adk_tools_to_openai([]) is None


# ---- Response schema conversion -------------------------------------------


class TestResponseSchemaConversion:
    """Pydantic model / dict → json_schema format."""

    def test_dict_schema(self):
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        result = _response_schema_to_format(schema)
        assert result is not None
        assert result["type"] == "json_schema"
        assert result["json_schema"]["strict"] is True
        assert result["json_schema"]["schema"]["additionalProperties"] is False

    def test_none_returns_none(self):
        assert _response_schema_to_format(None) is None

    def test_passthrough_json_schema(self):
        """Already-structured dict should pass through."""
        existing = {"type": "json_schema", "json_schema": {"name": "x", "schema": {}}}
        result = _response_schema_to_format(existing)
        assert result == existing


# ---- Config param mapping --------------------------------------------------


class TestConfigParamMapping:
    """temperature, max_output_tokens rename, stop_sequences rename."""

    def test_temperature(self):
        config = types.GenerateContentConfig(temperature=0.7)
        params = _config_to_params(config)
        assert params["temperature"] == 0.7

    def test_max_output_tokens_renamed(self):
        config = types.GenerateContentConfig(max_output_tokens=1000)
        params = _config_to_params(config)
        assert "max_completion_tokens" in params
        assert "max_output_tokens" not in params

    def test_stop_sequences_renamed(self):
        config = types.GenerateContentConfig(stop_sequences=["END"])
        params = _config_to_params(config)
        assert params["stop"] == ["END"]
        assert "stop_sequences" not in params

    def test_empty_config(self):
        config = types.GenerateContentConfig()
        params = _config_to_params(config)
        assert params == {}


# ---- Finish reason mapping -------------------------------------------------


class TestFinishReasonMapping:
    """All cases: stop, length, tool_calls, content_filter."""

    def test_stop(self):
        assert _map_finish_reason("stop") == types.FinishReason.STOP

    def test_length(self):
        assert _map_finish_reason("length") == types.FinishReason.MAX_TOKENS

    def test_tool_calls(self):
        assert _map_finish_reason("tool_calls") == types.FinishReason.STOP

    def test_content_filter(self):
        assert _map_finish_reason("content_filter") == types.FinishReason.SAFETY

    def test_none(self):
        assert _map_finish_reason(None) == types.FinishReason.STOP

    def test_unknown(self):
        assert _map_finish_reason("something_else") == types.FinishReason.OTHER


# ---- Response → LlmResponse -----------------------------------------------


class TestResponseToLlmResponse:
    """ChatCompletion → LlmResponse with text, tool calls, usage."""

    def _make_response(self, *, content=None, tool_calls=None, finish_reason="stop"):
        """Build a minimal mock ChatCompletion-like response."""
        message = SimpleNamespace(
            content=content,
            tool_calls=tool_calls,
        )
        choice = SimpleNamespace(
            message=message,
            finish_reason=finish_reason,
        )
        usage = SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
        )
        return SimpleNamespace(
            choices=[choice],
            usage=usage,
            model="test-model",
        )

    def test_text_response(self):
        resp = self._make_response(content="Hello world")
        llm_resp = _openai_response_to_llm_response(resp)
        assert llm_resp.partial is False
        assert llm_resp.content is not None
        assert llm_resp.content.parts[0].text == "Hello world"
        assert llm_resp.usage_metadata is not None
        assert llm_resp.usage_metadata.prompt_token_count == 10

    def test_tool_call_response(self):
        tc = SimpleNamespace(
            type="function",
            id="call_1",
            function=SimpleNamespace(
                name="search",
                arguments='{"q": "test"}',
            ),
        )
        resp = self._make_response(tool_calls=[tc], finish_reason="tool_calls")
        llm_resp = _openai_response_to_llm_response(resp)
        assert llm_resp.content is not None
        parts = llm_resp.content.parts
        assert any(p.function_call for p in parts)


# ---- Ensure tool results --------------------------------------------------


class TestEnsureToolResults:
    """Missing tool results get placeholder messages."""

    def test_heals_missing_result(self):
        messages = [
            {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function"}]},
            {"role": "user", "content": "continue"},
        ]
        healed = _ensure_tool_results(messages)
        # Should have: assistant, tool(placeholder), user
        assert len(healed) == 3
        assert healed[1]["role"] == "tool"
        assert healed[1]["tool_call_id"] == "c1"

    def test_no_healing_needed(self):
        messages = [
            {"role": "assistant", "tool_calls": [{"id": "c1"}]},
            {"role": "tool", "tool_call_id": "c1", "content": "result"},
            {"role": "user", "content": "ok"},
        ]
        healed = _ensure_tool_results(messages)
        assert len(healed) == 3

    def test_empty_messages(self):
        assert _ensure_tool_results([]) == []

    def test_trailing_missing_result(self):
        messages = [
            {"role": "assistant", "tool_calls": [{"id": "c2"}]},
        ]
        healed = _ensure_tool_results(messages)
        assert len(healed) == 2
        assert healed[1]["role"] == "tool"


# ---- Schema to dict -------------------------------------------------------


class TestSchemaToDict:
    """types.Schema → plain dict recursive conversion."""

    def test_basic_schema(self):
        schema = types.Schema(
            type=types.Type.OBJECT,
            properties={
                "name": types.Schema(type=types.Type.STRING),
            },
        )
        result = _schema_to_dict(schema)
        assert result["type"] == "object"
        assert result["properties"]["name"]["type"] == "string"

    def test_dict_passthrough(self):
        d = {"type": "string"}
        result = _schema_to_dict(d)
        assert result == {"type": "string"}

    def test_array_with_items(self):
        schema = types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(type=types.Type.INTEGER),
        )
        result = _schema_to_dict(schema)
        assert result["type"] == "array"
        assert result["items"]["type"] == "integer"
