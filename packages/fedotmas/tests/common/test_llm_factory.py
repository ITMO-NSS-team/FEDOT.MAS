"""Factory tests — make_llm returns LiteLlm with correct transport."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest
from fedotmas._settings import ModelConfig, resolve_model_config
from fedotmas.common.llm import (
    _ERROR_PAYLOAD_LEN,
    LLMRequestTimeout,
    _invalid_tool_argument_names,
    _ProxyClient,
    llm_request_deadline,
    make_llm,
)
from fedotmas.mas.builder import build_routing_system
from fedotmas.mas.models import MASConfig
from google.adk.models.lite_llm import LiteLlm, _function_declaration_to_tool_param
from pydantic import BaseModel


class _ErrorResponse(BaseModel):
    choices: list[dict[str, str]]
    detail: str = ""


class _SingleChunkStream:
    def __init__(self, chunk):
        self.chunk = chunk

    async def __anext__(self):
        return self.chunk


def _client_with_response(response):
    client = _ProxyClient("http://localhost:9090/v1", "test", None)
    client._client = MagicMock()
    client._client.chat.completions.create = AsyncMock(return_value=response)
    return client


def _response(finish_reason: str = "stop"):
    response = MagicMock()
    response.choices = [{"finish_reason": finish_reason}]
    response.model_dump.return_value = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "openai/gpt-oss-120b",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "done"},
                "finish_reason": finish_reason,
            }
        ],
    }
    return response


def _tool_response(arguments: str):
    response = _response("tool_calls")
    response.model_dump.return_value["choices"][0]["message"] = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "example_tool", "arguments": arguments},
            }
        ],
    }
    return response


class TestMakeLlm:
    def test_creates_litellm_with_model(self):
        cfg = ModelConfig(model="openrouter/meta-llama/llama-3-70b")
        llm = make_llm(cfg)
        assert isinstance(llm, LiteLlm)
        assert llm.model == "openrouter/meta-llama/llama-3-70b"

    def test_passes_api_key(self):
        cfg = ModelConfig(model="gpt-4o", api_key="sk-test")
        llm = make_llm(cfg)
        assert llm._additional_args.get("api_key") == "sk-test"

    def test_no_extra_kwargs_when_defaults(self):
        cfg = ModelConfig(model="openai/gpt-4o")
        llm = make_llm(cfg)
        assert "api_base" not in llm._additional_args
        assert "api_key" not in llm._additional_args
        assert "extra_body" not in llm._additional_args

    def test_openai_model(self):
        cfg = ModelConfig(model="openai/gpt-4o")
        llm = make_llm(cfg)
        assert isinstance(llm, LiteLlm)
        assert llm.model == "openai/gpt-4o"

    def test_direct_mode_no_proxy_client(self):
        cfg = ModelConfig(model="openai/gpt-4o")
        llm = make_llm(cfg)
        assert not isinstance(llm.llm_client, _ProxyClient)

    def test_direct_mode_passes_extra_body(self):
        extra_body = {"provider": {"ignore": ["Azure"]}}
        cfg = ModelConfig(model="openai/gpt-4o", extra_body=extra_body)
        llm = make_llm(cfg)
        assert llm._additional_args["extra_body"] == extra_body


class TestProxyMode:
    def test_sets_proxy_client(self):
        cfg = ModelConfig(
            model="openrouter/openai/gpt-4o",
            api_base="http://localhost:9090/litellm",
        )
        llm = make_llm(cfg)
        assert isinstance(llm.llm_client, _ProxyClient)

    def test_empty_additional_args(self):
        cfg = ModelConfig(
            model="openrouter/openai/gpt-4o",
            api_base="http://localhost:9090/litellm",
        )
        llm = make_llm(cfg)
        assert not llm._additional_args

    def test_preserves_model(self):
        cfg = ModelConfig(
            model="openrouter/openai/gpt-4o",
            api_base="http://localhost:9090/litellm",
        )
        llm = make_llm(cfg)
        assert llm.model == "openrouter/openai/gpt-4o"

    def test_default_api_key(self):
        cfg = ModelConfig(
            model="openai/gpt-4o",
            api_base="http://localhost:9090/litellm",
        )
        llm = make_llm(cfg)
        assert isinstance(llm.llm_client, _ProxyClient)
        assert llm.llm_client._client.api_key == "no-key"

    def test_custom_api_key(self):
        cfg = ModelConfig(
            model="openai/gpt-4o",
            api_base="http://localhost:9090/litellm",
            api_key="sk-proxy-key",
        )
        llm = make_llm(cfg)
        assert llm.llm_client._client.api_key == "sk-proxy-key"

    def test_base_url_set(self):
        cfg = ModelConfig(
            model="openai/gpt-4o",
            api_base="http://localhost:9090/litellm",
        )
        llm = make_llm(cfg)
        assert str(llm.llm_client._client.base_url) == "http://localhost:9090/litellm/"

    def test_proxy_client_gets_extra_body(self):
        extra_body = {"provider": {"only": ["Chutes"], "allow_fallbacks": False}}
        cfg = ModelConfig(
            model="openai/gpt-4o",
            api_base="http://localhost:9090/litellm",
            extra_body=extra_body,
        )
        llm = make_llm(cfg)
        assert llm.llm_client._extra_body == extra_body


class TestResolveModelConfig:
    def test_picks_up_env_base_url(self, monkeypatch):
        monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:9090/litellm")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        cfg = resolve_model_config("openrouter/openai/gpt-4o")
        assert cfg.api_base == "http://localhost:9090/litellm"
        assert cfg.api_key is None

    def test_picks_up_env_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        cfg = resolve_model_config("openai/gpt-4o")
        assert cfg.api_key == "sk-test-key"
        assert cfg.api_base is None

    def test_no_env_defaults_to_none(self, monkeypatch):
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        cfg = resolve_model_config("openai/gpt-4o")
        assert cfg.api_base is None
        assert cfg.api_key is None

    def test_picks_up_provider_routing_env(self, monkeypatch):
        monkeypatch.setenv("FEDOTMAS_PROVIDER_IGNORE", "Azure,OpenAI")
        monkeypatch.setenv("FEDOTMAS_PROVIDER_ALLOW_FALLBACKS", "false")
        monkeypatch.setenv("FEDOTMAS_PROVIDER_SORT", "throughput")
        cfg = resolve_model_config("openrouter/qwen/qwen-3.6-finetuned")
        assert cfg.extra_body == {
            "provider": {
                "ignore": ["Azure", "OpenAI"],
                "allow_fallbacks": False,
                "sort": "throughput",
            }
        }

    def test_picks_up_provider_sort_partition_env(self, monkeypatch):
        monkeypatch.setenv("FEDOTMAS_PROVIDER_SORT_BY", "throughput")
        monkeypatch.setenv("FEDOTMAS_PROVIDER_SORT_PARTITION", "none")
        cfg = resolve_model_config("openrouter/qwen/qwen-3.6-finetuned")
        assert cfg.extra_body == {
            "provider": {
                "sort": {
                    "by": "throughput",
                    "partition": "none",
                }
            }
        }

    def test_explicit_modelconfig_unchanged(self):
        original = ModelConfig(
            model="openai/gpt-4o",
            api_base="http://custom:8080",
            api_key="sk-custom",
            extra_body={"provider": {"only": ["Azure"]}},
        )
        result = resolve_model_config(original)
        assert result is original


class TestProxyClientToolCompatibility:
    """ADK single-turn agent tools must be valid OpenAI-compatible tools."""

    async def test_forwards_actual_single_turn_worker_tool_schema(self):
        client = _client_with_response(_response())
        config = MASConfig(
            coordinator={
                "name": "coordinator",
                "description": "Coordinates",
                "instruction": "Route.",
            },
            workers=[
                {
                    "name": "worker1",
                    "description": "First worker",
                    "instruction": "Work.",
                },
                {
                    "name": "worker2",
                    "description": "Second worker",
                    "instruction": "Work.",
                },
            ],
        )
        coordinator = build_routing_system(config, autonomous=False)
        # Intentional canary: ADK exposes this conversion only as a private API.
        adk_worker_tools = [
            _function_declaration_to_tool_param(tool._get_declaration())
            for tool in await coordinator.canonical_tools()
        ]

        await client.acompletion(
            "openai/gpt-oss-120b",
            [{"role": "user", "content": "run"}],
            adk_worker_tools,
        )

        payload = client._client.chat.completions.create.await_args.kwargs["tools"]
        assert payload == adk_worker_tools
        assert [tool["function"]["name"] for tool in payload] == ["worker1", "worker2"]
        parameters = payload[0]["function"]["parameters"]
        assert parameters["type"] == "object"
        assert parameters["properties"]["request"]["type"] == "string"
        assert parameters["required"] == ["request"]


class TestProxyClientErrors:
    async def test_error_finish_reason_raises(self):
        client = _client_with_response(_response("error"))

        with pytest.raises(RuntimeError, match="finish_reason='error'"):
            await client.acompletion(
                "openai/gpt-oss-120b", [{"role": "user", "content": "run"}], []
            )

    async def test_stream_error_finish_reason_raises(self):
        chunk = _ErrorResponse(choices=[{"finish_reason": "error"}])
        client = _client_with_response(_SingleChunkStream(chunk))

        stream = await client.acompletion(
            "openai/gpt-oss-120b",
            [{"role": "user", "content": "run"}],
            [],
            stream=True,
        )

        with pytest.raises(RuntimeError, match="finish_reason='error'"):
            await anext(stream)

    async def test_error_finish_reason_payload_is_truncated(self):
        response = _ErrorResponse(
            choices=[{"finish_reason": "error"}], detail="x" * 3000
        )
        client = _client_with_response(response)

        with pytest.raises(RuntimeError) as exc_info:
            await client.acompletion(
                "openai/gpt-oss-120b", [{"role": "user", "content": "run"}], []
            )

        message = str(exc_info.value)
        assert '"detail": "' in message
        assert message.endswith("... (truncated)")
        assert len(message) <= (
            len("LLM provider returned finish_reason='error': ")
            + _ERROR_PAYLOAD_LEN
            + len("... (truncated)")
        )


class TestProxyClientToolArgumentValidation:
    def test_text_completion_with_no_tool_calls_is_valid(self):
        assert (
            _invalid_tool_argument_names(
                {"choices": [{"message": {"role": "assistant", "tool_calls": None}}]}
            )
            == []
        )

    def test_empty_tool_calls_are_valid(self):
        assert (
            _invalid_tool_argument_names(
                {"choices": [{"message": {"role": "assistant", "tool_calls": []}}]}
            )
            == []
        )

    def test_missing_or_empty_choices_are_valid(self):
        assert _invalid_tool_argument_names({"choices": None}) == []
        assert _invalid_tool_argument_names({"choices": []}) == []
        assert _invalid_tool_argument_names({}) == []

    def test_valid_and_invalid_tool_arguments_are_distinguished(self):
        assert _invalid_tool_argument_names(_tool_response('{"value": 1}')) == []
        assert _invalid_tool_argument_names(_tool_response("{")) == ["example_tool"]

    async def test_retries_malformed_tool_arguments_then_returns_valid_response(self):
        client = _client_with_response(_tool_response('{"broken":'))
        valid = _tool_response('{"value": 1}')
        client._client.chat.completions.create.side_effect = [
            _tool_response('{"broken":'),
            valid,
        ]

        result = await client.acompletion(
            "openai/test", [{"role": "user", "content": "x"}], []
        )

        assert (
            result.choices[0].message.tool_calls[0].function.arguments == '{"value": 1}'
        )
        assert client._client.chat.completions.create.await_count == 2
        retry = client._client.chat.completions.create.await_args_list[1].kwargs
        assert retry["messages"][-1]["content"].startswith("The previous tool-call")
        assert "max_tokens" not in retry

    async def test_explicit_output_limit_is_preserved(self):
        client = _client_with_response(_response())
        await client.acompletion(
            "openai/test", [{"role": "user", "content": "x"}], [], max_tokens=1234
        )
        assert (
            client._client.chat.completions.create.await_args.kwargs["max_tokens"]
            == 1234
        )

    async def test_usage_from_malformed_argument_retry_is_aggregated(self):
        malformed = _tool_response("{")
        malformed.model_dump.return_value["usage"] = {
            "prompt_tokens": 11,
            "completion_tokens": 5,
            "total_tokens": 16,
        }
        valid = _tool_response('{"ok": true}')
        valid.model_dump.return_value["usage"] = {
            "prompt_tokens": 7,
            "completion_tokens": 3,
            "total_tokens": 10,
        }
        client = _client_with_response(malformed)
        client._client.chat.completions.create.side_effect = [malformed, valid]

        response = await client.acompletion(
            "openai/test", [{"role": "user", "content": "x"}], []
        )

        assert response.usage.prompt_tokens == 18
        assert response.usage.completion_tokens == 8
        assert response.usage.total_tokens == 26

    async def test_usage_survives_transport_error_during_retry(self):
        malformed = _tool_response("{")
        malformed.model_dump.return_value["usage"] = {
            "prompt_tokens": 11,
            "completion_tokens": 5,
            "total_tokens": 16,
        }
        transport_error = RuntimeError("transport retry failed")
        client = _client_with_response(malformed)
        client._client.chat.completions.create.side_effect = [
            malformed,
            transport_error,
        ]

        with pytest.raises(RuntimeError, match="transport retry failed") as raised:
            await client.acompletion(
                "openai/test", [{"role": "user", "content": "x"}], []
            )

        assert raised.value.prompt_tokens == 11
        assert raised.value.completion_tokens == 5

    async def test_repeated_malformed_tool_arguments_fail_clearly(self):
        client = _client_with_response(_tool_response("{"))
        client._client.chat.completions.create.side_effect = [_tool_response("{")] * 3

        with pytest.raises(RuntimeError, match="malformed JSON tool arguments"):
            await client.acompletion(
                "openai/test", [{"role": "user", "content": "x"}], []
            )

        assert client._client.chat.completions.create.await_count == 3

    async def test_valid_tool_arguments_are_returned_unchanged(self):
        client = _client_with_response(_tool_response('{"value": 1}'))

        result = await client.acompletion(
            "openai/test", [{"role": "user", "content": "x"}], []
        )

        assert (
            result.choices[0].message.tool_calls[0].function.arguments == '{"value": 1}'
        )
        assert client._client.chat.completions.create.await_count == 1

    async def test_text_completion_with_no_tool_calls_returns_normally(self):
        response = _response()
        response.model_dump.return_value["choices"][0]["message"] = {
            "role": "assistant",
            "content": "done",
            "tool_calls": None,
        }
        client = _client_with_response(response)

        result = await client.acompletion(
            "openai/test", [{"role": "user", "content": "x"}], []
        )

        assert result.choices[0].message.content == "done"


def _timeout_client(monkeypatch, seconds="0.02"):
    monkeypatch.setenv("FEDOTMAS_LLM_REQUEST_TIMEOUT_S", seconds)
    return _client_with_response(_response())


class TestProxyClientRequestTimeout:
    async def test_hung_initial_request_times_out_with_stable_code(self, monkeypatch):
        client = _timeout_client(monkeypatch)

        async def hang(**kwargs):
            await asyncio.sleep(1)

        client._client.chat.completions.create.side_effect = hang
        started = time.monotonic()
        with pytest.raises(LLMRequestTimeout, match="LLM_REQUEST_TIMEOUT") as raised:
            await client.acompletion("openai/test", [{"role": "user", "content": "x"}], [])
        assert time.monotonic() - started < 0.5
        assert raised.value.code == "LLM_REQUEST_TIMEOUT"
        assert client._client.chat.completions.create.await_count == 2

    async def test_timeout_then_one_retry_succeeds(self, monkeypatch):
        client = _timeout_client(monkeypatch)
        valid = _response()

        async def timeout_once(**kwargs):
            if client._client.chat.completions.create.await_count == 1:
                await asyncio.sleep(1)
            return valid

        client._client.chat.completions.create.side_effect = timeout_once
        result = await client.acompletion("openai/test", [{"role": "user", "content": "x"}], [])
        assert result.choices[0].message.content == "done"
        assert client._client.chat.completions.create.await_count == 2

    async def test_two_timeouts_raise_after_one_retry(self, monkeypatch):
        client = _timeout_client(monkeypatch, "0.01")

        async def hang(**kwargs):
            await asyncio.sleep(1)

        client._client.chat.completions.create.side_effect = hang
        with pytest.raises(LLMRequestTimeout, match="LLM_REQUEST_TIMEOUT"):
            await client.acompletion("openai/test", [{"role": "user", "content": "x"}], [])
        assert client._client.chat.completions.create.await_count == 2

    async def test_malformed_argument_retry_is_bounded_and_keeps_usage(self, monkeypatch):
        client = _timeout_client(monkeypatch, "0.01")
        malformed = _tool_response("{")
        malformed.model_dump.return_value["usage"] = {
            "prompt_tokens": 11, "completion_tokens": 5, "total_tokens": 16,
        }

        async def responses(**kwargs):
            if client._client.chat.completions.create.await_count == 1:
                return malformed
            await asyncio.sleep(1)

        client._client.chat.completions.create.side_effect = responses
        with pytest.raises(LLMRequestTimeout) as raised:
            await client.acompletion("openai/test", [{"role": "user", "content": "x"}], [])
        assert client._client.chat.completions.create.await_count == 3
        assert raised.value.prompt_tokens == 11
        assert raised.value.completion_tokens == 5

    def test_invalid_timeout_setting_uses_default(self, monkeypatch):
        for value in ("invalid", "0", "-1", "inf", "nan"):
            client = _timeout_client(monkeypatch, value)
            assert client._request_timeout == 120

    async def test_fast_request_behavior_is_unchanged(self, monkeypatch):
        client = _timeout_client(monkeypatch)
        result = await client.acompletion("openai/test", [{"role": "user", "content": "x"}], [])
        assert result.choices[0].message.content == "done"
        request_kwargs = client._client.chat.completions.create.await_args.kwargs
        assert request_kwargs["timeout"] == pytest.approx(0.02)

    async def test_task_deadline_caps_request_timeout(self, monkeypatch):
        client = _timeout_client(monkeypatch, "10")
        with llm_request_deadline(time.monotonic() + 0.05):
            await client.acompletion(
                "openai/test", [{"role": "user", "content": "x"}], []
            )
        request_timeout = client._client.chat.completions.create.await_args.kwargs[
            "timeout"
        ]
        assert request_timeout < 0.05

    async def test_stream_chunk_wait_is_bounded(self, monkeypatch):
        client = _timeout_client(monkeypatch, "0.01")

        class HangingStream:
            async def __anext__(self):
                await asyncio.sleep(1)

        client._client.chat.completions.create.return_value = HangingStream()
        stream = await client.acompletion(
            "openai/test", [{"role": "user", "content": "x"}], [], stream=True
        )
        with pytest.raises(LLMRequestTimeout, match="LLM_REQUEST_TIMEOUT"):
            await anext(stream)
