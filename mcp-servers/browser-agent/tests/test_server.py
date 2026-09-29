from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from mcp_browser_agent.server import (
    BrowserTaskResult,
    _history_result,
    _is_setup_error,
    _llm_settings,
    _partial_usage,
    _run_browser_task,
    _track_llm_calls,
    mcp,
)


class _FakeHistory:
    def __init__(
        self,
        findings="Answer: 42",
        urls=None,
        errors=None,
        steps=2,
        done=True,
        success=True,
    ):
        self._findings = findings
        self._urls = urls or ["https://example.org/page"]
        self._errors = errors or []
        self._steps = steps
        self._done = done
        self._success = success
        self.usage = SimpleNamespace(
            total_prompt_tokens=100,
            total_completion_tokens=20,
            total_tokens=120,
            entry_count=3,
        )

    def is_done(self):
        return self._done

    def is_successful(self):
        return self._success

    def final_result(self):
        return self._findings

    def urls(self):
        return self._urls

    def errors(self):
        return self._errors

    def total_duration_seconds(self):
        return 1.25

    def __len__(self):
        return self._steps


def test_server_exposes_only_high_level_task_tool():
    async def list_tools():
        return await mcp.list_tools()

    tools = asyncio.run(list_tools())
    assert [tool.name for tool in tools] == ["complete_browser_task"]
    assert "multi-step" in tools[0].description


def test_history_result_is_compact_and_drops_screenshot_data():
    screenshot = "A" * 2_000
    result = _history_result(
        _FakeHistory(
            findings=f"Found it. data:image/png;base64,{screenshot}",
            urls=["https://example.org/page", "file:///tmp/screenshot.png"],
            errors=["warning " + screenshot],
        )
    )

    assert result.status == "completed"
    assert result.relevant_urls == ["https://example.org/page"]
    assert "data:image/png;base64" not in result.findings
    assert screenshot not in result.findings
    assert screenshot not in result.errors[0]
    assert result.steps_taken == 2
    assert result.duration_seconds == 1.25


def test_history_without_findings_is_incomplete():
    result = _history_result(
        _FakeHistory(findings="", steps=3, done=False, success=None)
    )

    assert result.status == "incomplete"
    assert result.errors == ["Browser-Use stopped without confirmed completion"]


def test_missing_llm_key_returns_blocked_result(monkeypatch):
    for name in (
        "BROWSER_AGENT_API_KEY",
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)

    result = asyncio.run(_run_browser_task("find an answer", 3))

    assert isinstance(result, BrowserTaskResult)
    assert result.status == "blocked"
    assert "OPENAI_API_KEY" in result.errors[0]


def test_browser_task_uses_internal_agent_and_cleans_up_browser(monkeypatch):
    import sys
    from types import ModuleType

    calls = {}

    class FakeBrowser:
        def __init__(self, **kwargs):
            calls["browser_options"] = kwargs

        async def kill(self):
            calls["closed"] = True

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            calls["llm_options"] = kwargs

    class FakeAgent:
        def __init__(self, **kwargs):
            calls["agent_options"] = kwargs

        async def run(self, *, max_steps):
            calls["max_steps"] = max_steps
            return _FakeHistory()

    fake_browser_use = ModuleType("browser_use")
    fake_browser_use.Agent = FakeAgent
    fake_browser_use.Browser = FakeBrowser
    fake_browser_use.ChatOpenAI = FakeChatOpenAI
    monkeypatch.setitem(sys.modules, "browser_use", fake_browser_use)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    result = asyncio.run(_run_browser_task("find an answer", 4))

    assert result.status == "completed"
    assert result.findings == "Answer: 42"
    assert calls["max_steps"] == 4
    assert calls["browser_options"] == {
        "headless": True,
        "user_data_dir": None,
        "keep_alive": False,
    }
    assert calls["agent_options"]["task"] == "find an answer"
    assert calls["closed"] is True
    assert calls["agent_options"]["calculate_cost"] is True
    assert result.usage.total_tokens == 120


def test_llm_settings_use_openrouter_when_configured(monkeypatch):
    from mcp_browser_agent.server import _llm_settings

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("BROWSER_AGENT_MODEL", raising=False)
    monkeypatch.delenv("BROWSER_AGENT_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    assert _llm_settings() == (
        "openai/gpt-4o-mini",
        "test-key",
        "https://openrouter.ai/api/v1",
    )


@pytest.fixture(autouse=True)
def clean_provider_environment(monkeypatch):
    for prefix in ("BROWSER_AGENT", "FEDOTMAS_GAIA_WORKER", "OPENAI", "OPENROUTER"):
        for suffix in ("MODEL", "API_KEY", "BASE_URL"):
            monkeypatch.delenv(f"{prefix}_{suffix}", raising=False)


@pytest.mark.parametrize(
    "done,success,status",
    [
        (True, True, "completed"),
        (True, False, "failed"),
        (False, None, "incomplete"),
        (True, None, "incomplete"),
    ],
)
def test_status_uses_completion_flags_even_with_nonempty_findings(
    done, success, status
):
    result = _history_result(_FakeHistory(done=done, success=success))
    assert result.status == status
    assert result.findings == "Answer: 42"
    assert result.usage.model_dump() == {
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "total_tokens": 120,
        "llm_invocations": 3,
    }


def test_openrouter_does_not_use_openai_endpoint(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openai.example/v1")
    assert _llm_settings() == (
        "openai/gpt-4o-mini",
        "router-key",
        "https://openrouter.ai/api/v1",
    )


def test_scoped_provider_precedence(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-key")
    for prefix, model, key, base in (
        (
            "FEDOTMAS_GAIA_WORKER",
            "worker-model",
            "worker-key",
            "https://worker.example/v1",
        ),
        ("BROWSER_AGENT", "browser-model", "browser-key", "https://browser.example/v1"),
    ):
        for suffix, value in (("MODEL", model), ("API_KEY", key), ("BASE_URL", base)):
            monkeypatch.setenv(f"{prefix}_{suffix}", value)
        assert _llm_settings() == (model, key, base)


def test_custom_endpoint_without_matching_key_is_blocked(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-key")
    monkeypatch.setenv("BROWSER_AGENT_BASE_URL", "https://custom.example/v1")
    result = asyncio.run(_run_browser_task("task", 1))
    assert result.status == "blocked"
    assert "BROWSER_AGENT_API_KEY" in result.errors[0]
    assert "router-key" not in result.model_dump_json()


def test_missing_local_chromium_is_setup_blocker():
    assert _is_setup_error(
        RuntimeError(
            "No local Chrome/Chromium install found, and failed to install with playwright"
        )
    )


def test_model_only_override_preserves_provider_pair(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-key")
    monkeypatch.setenv("BROWSER_AGENT_MODEL", "vendor/custom-model")
    assert _llm_settings() == (
        "vendor/custom-model",
        "router-key",
        "https://openrouter.ai/api/v1",
    )


def test_browser_key_only_uses_openai_endpoint_even_with_router_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-key")
    monkeypatch.setenv("BROWSER_AGENT_API_KEY", "browser-key")
    assert _llm_settings() == (
        "gpt-4o-mini",
        "browser-key",
        "https://api.openai.com/v1",
    )


def test_worker_key_only_uses_openrouter_endpoint_even_with_openai_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openai.example/v1")
    monkeypatch.setenv("FEDOTMAS_GAIA_WORKER_API_KEY", "worker-key")
    assert _llm_settings() == (
        "openai/gpt-4o-mini",
        "worker-key",
        "https://openrouter.ai/api/v1",
    )


@pytest.mark.parametrize("status", ["completed", "failed", "incomplete", "blocked"])
def test_mcp_error_flag_and_payload_preserve_usage(monkeypatch, status):
    from mcp_browser_agent import server

    async def run(task, max_steps):
        result = _history_result(_FakeHistory())
        result.status = status
        return result

    monkeypatch.setattr(server, "_run_browser_task", run)
    result = asyncio.run(mcp.call_tool("complete_browser_task", {"task": "task"}))
    assert result.is_error is (status != "completed")
    assert result.structured_content["usage"]["total_tokens"] == 120
    if status != "completed":
        assert (
            result.structured_content["error_code"] == f"BROWSER_AGENT_{status.upper()}"
        )
    assert json.loads(result.content[0].text)["status"] == status


def test_partial_usage_preserves_tokens_after_exception():
    agent = SimpleNamespace(
        token_cost_service=SimpleNamespace(
            usage_history=[
                SimpleNamespace(
                    usage=SimpleNamespace(prompt_tokens=11, completion_tokens=3)
                ),
                SimpleNamespace(
                    usage=SimpleNamespace(prompt_tokens=15, completion_tokens=4)
                ),
            ]
        )
    )
    assert _partial_usage(agent).model_dump() == {
        "prompt_tokens": 26,
        "completion_tokens": 7,
        "total_tokens": 33,
        "llm_invocations": 2,
    }
    assert _partial_usage(None).total_tokens is None


def test_llm_invocation_counter_includes_calls_without_usage():
    class FakeLLM:
        async def ainvoke(self, *args, **kwargs):
            return SimpleNamespace(usage=None)

    llm = FakeLLM()
    agent = SimpleNamespace(
        token_cost_service=SimpleNamespace(registered_llms={"one": llm})
    )
    count = _track_llm_calls(agent)
    asyncio.run(llm.ainvoke("task"))
    assert count == [1]
