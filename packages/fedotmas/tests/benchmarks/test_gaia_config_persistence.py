from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fedotmas import ModelConfig
from fedotmas.core.runner import PipelineExecutionError, PipelineResult
from fedotmas.maw.models import MAWConfig
from fedotmas.plugins import ResearchTelemetry

from benchmarks.gaia.run_gaia import (
    GAIA_BASE_MCP_SERVERS,
    _attempt_diagnostics,
    _gaia_mcp_registry,
    _gaia_mcp_servers,
    build_plugins,
    compute_token_summary,
    extract_answer_from_state,
    extract_terminal_answer,
    process_task,
    root_cause_summary,
)


def _config() -> MAWConfig:
    return MAWConfig.model_validate(
        {
            "agents": [
                {
                    "name": "researcher",
                    "instruction": "Research the task",
                    "output_key": "findings",
                    "tools": ["websearch-searxng"],
                    "model": "openai/gpt-4o",
                },
                {
                    "name": "answerer",
                    "instruction": "Answer from {findings}",
                    "output_key": "final_answer",
                    "model": "openai/gpt-4o",
                },
            ],
            "final_answer_agent": "answerer",
            "pipeline": {
                "type": "sequential",
                "children": [
                    {"type": "agent", "agent_name": "researcher"},
                    {"type": "agent", "agent_name": "answerer"},
                ],
            },
        }
    )


def test_gaia_default_mcp_servers_include_web_task_tools_and_light_sandbox(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("FEDOTMAS_GAIA_MCP_SERVERS", raising=False)
    monkeypatch.delenv("FEDOTMAS_GAIA_SEARCH_PROVIDERS", raising=False)
    monkeypatch.delenv("TAVILY_API_KEYS", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("E2B_API_KEY", raising=False)

    servers = _gaia_mcp_servers()

    assert servers == [
        *GAIA_BASE_MCP_SERVERS[:5],
        "sandbox-light",
        *GAIA_BASE_MCP_SERVERS[5:],
    ]
    assert "download" in servers
    assert "youtube-transcript" in servers
    assert "browser-agent" in servers
    assert "code-agent" in servers
    assert "research-controller" in servers

    registry = _gaia_mcp_registry(ModelConfig(model="openai/gpt-4o"))
    assert "research-controller" in registry
    assert "get_next_action" in registry["research-controller"].description


def test_gaia_adds_tavily_when_configured_and_supports_search_ab_modes(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("FEDOTMAS_GAIA_MCP_SERVERS", raising=False)
    monkeypatch.setenv("TAVILY_API_KEYS", " key-one, ,key-two ")
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("E2B_API_KEY", raising=False)
    monkeypatch.delenv("FEDOTMAS_GAIA_SEARCH_PROVIDERS", raising=False)

    defaults = _gaia_mcp_servers()
    assert "websearch-searxng" in defaults
    assert "websearch-tavily" in defaults
    assert "websearch-tavily" in _gaia_mcp_registry(ModelConfig(model="openai/gpt-4o"))

    for setting, expected in (
        ("searxng", {"websearch-searxng"}),
        ("tavily", {"websearch-tavily"}),
        ("searxng,tavily", {"websearch-searxng", "websearch-tavily"}),
    ):
        monkeypatch.setenv("FEDOTMAS_GAIA_SEARCH_PROVIDERS", setting)
        servers = _gaia_mcp_servers()
        assert expected <= set(servers)
        assert {"websearch-searxng", "websearch-tavily"} & set(servers) == expected


def test_gaia_uses_full_sandbox_when_e2b_key_is_set(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("FEDOTMAS_GAIA_MCP_SERVERS", raising=False)
    monkeypatch.setenv("E2B_API_KEY", "test-key")

    servers = _gaia_mcp_servers()

    assert "sandbox" in servers
    assert "sandbox-light" not in servers


def test_gaia_passes_resolved_worker_settings_to_browser_agent(monkeypatch):
    monkeypatch.delenv("FEDOTMAS_GAIA_MCP_SERVERS", raising=False)
    monkeypatch.setenv("E2B_API_KEY", "test-key")
    worker = ModelConfig(
        model="openai/gpt-6-luna",
        api_base="https://openrouter.ai/api/v1",
        api_key="worker-key",
    )

    browser = _gaia_mcp_registry(worker)["browser-agent"]

    assert browser.env["FEDOTMAS_GAIA_WORKER_MODEL"] == "openai/gpt-6-luna"
    assert (
        browser.env["FEDOTMAS_GAIA_WORKER_BASE_URL"] == "https://openrouter.ai/api/v1"
    )
    assert browser.env["FEDOTMAS_GAIA_WORKER_API_KEY"] == "worker-key"
    code_agent = _gaia_mcp_registry(worker)["code-agent"]
    assert code_agent.env["FEDOTMAS_GAIA_WORKER_MODEL"] == "openai/gpt-6-luna"
    assert code_agent.env["FEDOTMAS_GAIA_WORKER_API_KEY"] == "worker-key"
    assert (
        code_agent.env["FEDOTMAS_GAIA_WORKER_BASE_URL"]
        == "https://openrouter.ai/api/v1"
    )


def test_gaia_mcp_server_override_is_preserved(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FEDOTMAS_GAIA_MCP_SERVERS", "download, browser-agent")
    monkeypatch.setenv("E2B_API_KEY", "test-key")

    assert _gaia_mcp_servers() == ["download", "browser-agent"]


@pytest.mark.asyncio
async def test_gaia_browser_limit_is_per_agent_and_uses_budget_control(monkeypatch):
    monkeypatch.delenv("FEDOTMAS_GAIA_BROWSER_AGENT_LIMIT", raising=False)
    plugins = build_plugins(SimpleNamespace(), enable_langfuse=False)
    limit = next(
        plugin
        for plugin in plugins
        if getattr(plugin, "budget_kind", None) == "browser_agent"
    )
    telemetry = next(
        plugin for plugin in plugins if isinstance(plugin, ResearchTelemetry)
    )
    tool = SimpleNamespace(name="complete_browser_task", description="Browser task")

    def context(agent):
        return SimpleNamespace(
            _invocation_context=SimpleNamespace(
                session=SimpleNamespace(id="session"),
                agent=SimpleNamespace(name=agent),
            )
        )

    for index in range(3):
        assert (
            await limit.before_tool_callback(
                tool=tool,
                tool_args={"task": f"task {index}"},
                tool_context=context("researcher"),
            )
            is None
        )
    blocked = await limit.before_tool_callback(
        tool=tool, tool_args={"task": "task 4"}, tool_context=context("researcher")
    )
    sibling = await limit.before_tool_callback(
        tool=tool, tool_args={"task": "sibling"}, tool_context=context("sibling")
    )
    assert blocked["isError"] is True
    assert blocked["error_code"] == "WEB_BUDGET_EXHAUSTED"
    assert sibling is None
    assert telemetry.snapshot()["researcher"]["browser_agent_exhaustion"] == 1


def test_gaia_diagnostics_aggregate_browser_tokens_separately():
    records = [
        {
            "tokens": {"meta_prompt": 10},
            "research_telemetry": {
                "researcher": {
                    "browser_agent_prompt_tokens": 100,
                    "browser_agent_completion_tokens": 20,
                    "browser_agent_total_tokens": 120,
                    "browser_agent_llm_invocations": 2,
                    "browser_agent_steps": 4,
                }
            },
        },
        {
            "tokens": {"meta_prompt": 15},
            "research_telemetry": {
                "researcher": {
                    "browser_agent_prompt_tokens": 40,
                    "browser_agent_completion_tokens": 10,
                    "browser_agent_total_tokens": 50,
                    "browser_agent_llm_invocations": 1,
                    "browser_agent_steps": 3,
                }
            },
        },
    ]
    diagnostics = _attempt_diagnostics(records)
    summary = compute_token_summary([diagnostics])

    assert (
        diagnostics["research_telemetry"]["researcher"]["browser_agent_total_tokens"]
        == 170
    )
    assert summary["browser_agent"] == {
        "prompt_tokens": 140,
        "completion_tokens": 30,
        "total_tokens": 170,
        "llm_invocations": 3,
        "steps": 7,
        "usage_missing": 0,
    }
    assert summary["grand_total"]["prompt_tokens"] == 25


def test_gaia_reports_nested_code_agent_tokens_separately():
    summary = compute_token_summary(
        [
            {
                "tokens": {"pipeline_prompt": 100, "pipeline_completion": 20},
                "research_telemetry": {
                    "analyst": {
                        "code_agent_prompt_tokens": 30,
                        "code_agent_completion_tokens": 10,
                        "code_agent_total_tokens": 40,
                        "code_agent_llm_invocations": 2,
                        "code_agent_steps": 3,
                        "code_agent_cost_usd": 0.005,
                    }
                },
            }
        ]
    )

    assert summary["outer_worker_tokens"] == {
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "total_tokens": 120,
    }
    assert summary["code_agent_tokens"] == {
        "prompt_tokens": 30,
        "completion_tokens": 10,
        "total_tokens": 40,
    }
    assert summary["combined_tokens"]["total_tokens"] == 160
    assert summary["code_agent"]["llm_invocations"] == 2
    assert summary["code_agent"]["steps"] == 3
    assert summary["code_agent"]["cost_usd"] == 0.005


class _FakeMAW:
    def __init__(self, **_kwargs):
        self.generated_config = _config()
        self.last_result = SimpleNamespace(
            total_prompt_tokens=9, total_completion_tokens=4
        )
        self.meta_prompt_tokens = 3
        self.meta_completion_tokens = 2
        self.total_prompt_tokens = 12
        self.total_completion_tokens = 6
        self.elapsed = 1.5

    async def run(
        self,
        _query: str,
        *,
        timeout: int,
        final_answer_contract: str | None = None,
    ) -> dict[str, str]:
        assert timeout > 0
        assert "<solution>" not in _query
        assert "<solution>" in (final_answer_contract or "")
        return {"final_answer": "<solution>42</solution>"}


@pytest.mark.asyncio
async def test_successful_gaia_result_persists_serializable_generated_config(
    tmp_path: Path,
):
    task = SimpleNamespace(
        task_id="task-1",
        question="What is the answer?",
        ground_truth="42",
        file_path=None,
        file_name=None,
        difficulty="1",
    )
    benchmark = SimpleNamespace(is_correct_answer=lambda answer, truth: answer == truth)

    with patch("benchmarks.gaia.run_gaia.MAW", _FakeMAW):
        result = await process_task(task, benchmark, tmp_path, enable_langfuse=False)

    artifact = json.loads((tmp_path / "result.json").read_text())
    attempt = json.loads((tmp_path / "attempts" / "attempt_01.json").read_text())

    assert result["is_correct"] is True
    assert artifact["maw_config"] == _config().model_dump(mode="json")
    assert isinstance(artifact["research_telemetry"], dict)
    assert artifact["attempts"][0]["attempt_status"] == "succeeded"
    assert attempt["maw_config"] == _config().model_dump(mode="json")


@pytest.mark.asyncio
async def test_generated_config_survives_execution_failure(tmp_path: Path):
    class FailingMAW(_FakeMAW):
        def __init__(self, **kwargs):
            self._kwargs = kwargs
            super().__init__(**kwargs)
            self.last_result.state = {"findings": "partial evidence"}

        async def run(
            self,
            _query: str,
            *,
            timeout: int,
            final_answer_contract: str | None = None,
        ) -> dict[str, str]:
            assert "<solution>" in (final_answer_contract or "")
            telemetry = next(
                plugin
                for plugin in self._kwargs["plugins"]
                if plugin.__class__.__name__ == "ResearchTelemetry"
            )
            telemetry.attempt("researcher", "search", {"query": "partial query"})
            raise RuntimeError("execution failed after generation")

    task = SimpleNamespace(
        task_id="task-2",
        question="Question?",
        ground_truth="42",
        file_path=None,
        file_name=None,
        difficulty="1",
    )
    with (
        patch("benchmarks.gaia.run_gaia.MAW", FailingMAW),
        patch.dict("os.environ", {"FEDOTMAS_GAIA_TASK_ATTEMPTS": "1"}),
        pytest.raises(RuntimeError, match="execution failed"),
    ):
        await process_task(task, SimpleNamespace(), tmp_path, enable_langfuse=False)
    artifact = json.loads((tmp_path / "result.json").read_text())
    attempt = json.loads((tmp_path / "attempts" / "attempt_01.json").read_text())
    assert artifact["maw_config"] == _config().model_dump(mode="json")
    assert artifact["session_state"] == {"findings": "partial evidence"}
    assert artifact["tokens"]["meta_prompt"] == 3
    assert artifact["tokens"]["pipeline_prompt"] == 9
    assert artifact["root_cause"] == "exception.RuntimeError"
    assert artifact["research_telemetry"]["researcher"]["attempted_calls"] == 1
    assert attempt["attempt_status"] == "failed"
    assert isinstance(artifact["research_telemetry"], dict)


@pytest.mark.asyncio
async def test_retries_keep_each_attempt_diagnostics_and_sum_tokens(tmp_path: Path):
    class RetryMAW(_FakeMAW):
        calls = 0

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.plugins = kwargs["plugins"]
            type(self).calls += 1
            if type(self).calls == 1:
                self.last_result.state = {"findings": "first attempt partial"}

        async def run(
            self,
            _query: str,
            *,
            timeout: int,
            final_answer_contract: str | None = None,
        ) -> dict[str, str]:
            assert "<solution>" in (final_answer_contract or "")
            telemetry = next(
                plugin
                for plugin in self.plugins
                if plugin.__class__.__name__ == "ResearchTelemetry"
            )
            telemetry.attempt("researcher", "search", {"query": "retry query"})
            if type(self).calls == 1:
                raise RuntimeError("transient execution failure")
            return {"final_answer": "<solution>42</solution>"}

    task = SimpleNamespace(
        task_id="task-retry",
        question="Question?",
        ground_truth="42",
        file_path=None,
        file_name=None,
        difficulty="1",
    )
    benchmark = SimpleNamespace(is_correct_answer=lambda answer, truth: answer == truth)

    with (
        patch("benchmarks.gaia.run_gaia.MAW", RetryMAW),
        patch.dict("os.environ", {"FEDOTMAS_GAIA_TASK_ATTEMPTS": "2"}),
        patch("benchmarks.gaia.run_gaia.asyncio.sleep", new_callable=AsyncMock),
    ):
        result = await process_task(task, benchmark, tmp_path, enable_langfuse=False)

    attempt_one = json.loads((tmp_path / "attempts" / "attempt_01.json").read_text())
    attempt_two = json.loads((tmp_path / "attempts" / "attempt_02.json").read_text())
    assert [item["attempt_status"] for item in result["attempts"]] == [
        "failed",
        "succeeded",
    ]
    assert result["tokens"]["total_prompt"] == 24
    assert result["tokens"]["total_completion"] == 12
    assert result["elapsed"] == 3.0
    assert attempt_one["session_state"] == {"findings": "first attempt partial"}
    assert attempt_one["research_telemetry"]["researcher"]["unique_queries"] == 1
    assert attempt_two["attempt_status"] == "succeeded"
    assert result["research_telemetry"]["researcher"]["search_calls"] == 2
    assert result["research_telemetry"]["researcher"]["unique_queries"] == 1


def test_failed_results_are_included_in_token_summary():
    summary = compute_token_summary(
        [{"error": "execution failed", "tokens": {"meta_prompt": 7}}]
    )
    assert summary["meta_agent"]["prompt_tokens"] == 7


def test_pipeline_wrapper_preserves_underlying_root_cause():
    summary = root_cause_summary(
        PipelineExecutionError(RuntimeError("backend unavailable"), PipelineResult())
    )
    assert summary["last_exception"] == "RuntimeError"
    assert summary["message"] == "backend unavailable"
    assert summary["wrapper_exception"] == "PipelineExecutionError"


def test_gaia_answer_extraction_only_reads_the_configured_terminal_output():
    state = {
        "research_findings": "<solution>wrong intermediate value</solution>",
        "terminal_output": "Reasoning outside the tag. <solution>42</solution>",
    }

    assert extract_answer_from_state(state) == ""
    assert extract_terminal_answer(state, "terminal_output") == "42"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("partial_answer", "ground_truth"),
    [("<solution>wrong</solution>", "42"), ("<solution>42</solution>", "42")],
)
async def test_gaia_timeout_never_submits_intermediate_or_matching_partial_answer(
    tmp_path: Path, partial_answer: str, ground_truth: str
):
    class TimedOutMAW(_FakeMAW):
        async def run(
            self,
            _query: str,
            *,
            timeout: int,
            final_answer_contract: str | None = None,
        ) -> dict[str, str]:
            self.last_result = PipelineResult(
                state={"research_findings": partial_answer}, status="timed_out"
            )
            return self.last_result.state

    task = SimpleNamespace(
        task_id="task-timeout",
        question="Question?",
        ground_truth=ground_truth,
        file_path=None,
        file_name=None,
        difficulty="1",
    )
    scored: list[str] = []
    benchmark = SimpleNamespace(
        is_correct_answer=lambda answer, truth: scored.append(answer) or answer == truth
    )
    with (
        patch("benchmarks.gaia.run_gaia.MAW", TimedOutMAW),
        patch.dict("os.environ", {"FEDOTMAS_GAIA_TASK_ATTEMPTS": "1"}),
        pytest.raises(RuntimeError, match="status 'timed_out'"),
    ):
        await process_task(task, benchmark, tmp_path, enable_langfuse=False)

    artifact = json.loads((tmp_path / "result.json").read_text())
    attempt = json.loads((tmp_path / "attempts" / "attempt_01.json").read_text())
    assert scored == []
    assert artifact["pipeline_status"] == "timed_out"
    assert artifact["attempt_status"] == "incomplete"
    assert attempt["session_state"]["research_findings"] == partial_answer
    assert artifact["response"] == ""


@pytest.mark.asyncio
async def test_outer_gaia_timeout_is_recorded_as_incomplete(tmp_path: Path):
    class BackstopTimeoutMAW(_FakeMAW):
        async def run(
            self,
            _query: str,
            *,
            timeout: int,
            final_answer_contract: str | None = None,
        ) -> dict[str, str]:
            raise TimeoutError("outer execution backstop expired")

    task = SimpleNamespace(
        task_id="task-backstop-timeout",
        question="Question?",
        ground_truth="42",
        file_path=None,
        file_name=None,
        difficulty="1",
    )
    with (
        patch("benchmarks.gaia.run_gaia.MAW", BackstopTimeoutMAW),
        patch.dict("os.environ", {"FEDOTMAS_GAIA_TASK_ATTEMPTS": "1"}),
        pytest.raises(TimeoutError, match="backstop expired"),
    ):
        await process_task(task, SimpleNamespace(), tmp_path, enable_langfuse=False)

    artifact = json.loads((tmp_path / "result.json").read_text())
    assert artifact["pipeline_status"] == "timed_out"
    assert artifact["attempt_status"] == "incomplete"
    assert artifact["response"] == ""
