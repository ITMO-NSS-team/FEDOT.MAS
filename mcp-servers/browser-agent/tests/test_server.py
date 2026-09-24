from __future__ import annotations

import asyncio

from mcp_browser_agent.server import (
    BrowserTaskResult,
    _history_result,
    _run_browser_task,
    mcp,
)


class _FakeHistory:
    def __init__(self, findings="Answer: 42", urls=None, errors=None, steps=2):
        self._findings = findings
        self._urls = urls or ["https://example.org/page"]
        self._errors = errors or []
        self._steps = steps

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
    result = _history_result(_FakeHistory(findings="", steps=3))

    assert result.status == "incomplete"
    assert result.errors == ["Browser-Use stopped without returning findings"]


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
