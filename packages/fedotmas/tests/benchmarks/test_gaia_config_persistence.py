from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fedotmas import ModelConfig
from fedotmas.core.runner import PipelineExecutionError, PipelineResult
from fedotmas.maw.models import ArtifactRequirement, MAWConfig
from fedotmas.maw import builder as maw_builder
from fedotmas.plugins._code_agent_budget import CODE_AGENT_POLICY_STATE_KEY
from fedotmas.plugins import ResearchTelemetry

from benchmarks.gaia.run_gaia import (
    GAIA_BASE_MCP_SERVERS,
    _attempt_diagnostics,
    _gaia_max_agent_llm_turns,
    _gaia_mcp_registry,
    _gaia_mcp_servers,
    _load_frozen_config,
    _log_selected_models,
    _task_from_file,
    build_plugins,
    compute_metrics_by_level,
    compute_token_summary,
    extract_answer_from_state,
    extract_terminal_answer,
    _has_unresolved_answer_lineage,
    _matches_declared_answer_format,
    print_score_by_level,
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
    assert "code-agent" not in servers
    assert "sandbox-light" in servers
    assert "research-controller" in servers

    registry = _gaia_mcp_registry(ModelConfig(model="openai/gpt-4o"))
    assert "code-agent" not in registry
    assert "research-controller" in registry
    assert "get_next_action" in registry["research-controller"].description


def test_gaia_uses_one_tavily_search_interface_with_internal_fallback(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("FEDOTMAS_GAIA_MCP_SERVERS", raising=False)
    monkeypatch.setenv("TAVILY_API_KEYS", " key-one, ,key-two ")
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("E2B_API_KEY", raising=False)
    monkeypatch.delenv("FEDOTMAS_GAIA_SEARCH_PROVIDERS", raising=False)

    defaults = _gaia_mcp_servers()
    assert "websearch-tavily" in defaults
    assert "websearch-searxng" not in defaults
    assert "websearch-tavily" in _gaia_mcp_registry(ModelConfig(model="openai/gpt-4o"))
    assert "websearch-searxng" not in _gaia_mcp_registry(
        ModelConfig(model="openai/gpt-4o")
    )

    for setting in ("searxng", "tavily", "searxng,tavily"):
        monkeypatch.setenv("FEDOTMAS_GAIA_SEARCH_PROVIDERS", setting)
        servers = _gaia_mcp_servers()
        assert "websearch-tavily" in servers
        assert "websearch-searxng" not in servers


def test_gaia_uses_full_sandbox_when_e2b_key_is_set(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("FEDOTMAS_GAIA_MCP_SERVERS", raising=False)
    monkeypatch.setenv("E2B_API_KEY", "test-key")

    servers = _gaia_mcp_servers()

    assert "sandbox" in servers
    assert "sandbox-light" not in servers
    assert "code-agent" in servers


def test_gaia_mcp_override_cannot_enable_code_agent_without_e2b(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("FEDOTMAS_GAIA_MCP_SERVERS", "download, sandbox, code-agent")
    monkeypatch.delenv("E2B_API_KEY", raising=False)

    assert _gaia_mcp_servers() == ["download", "sandbox-light"]
    assert "code-agent" not in _gaia_mcp_registry(ModelConfig(model="openai/gpt-4o"))


def test_gaia_all_mcp_servers_excludes_e2b_servers_without_key(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("FEDOTMAS_GAIA_MCP_SERVERS", "all")
    monkeypatch.delenv("E2B_API_KEY", raising=False)

    servers = _gaia_mcp_servers()

    assert "sandbox-light" in servers
    assert not {"sandbox", "code-agent"} & set(servers)
    registry = _gaia_mcp_registry(ModelConfig(model="openai/gpt-4o"))
    assert "sandbox-light" in registry
    assert not {"sandbox", "code-agent"} & set(registry)


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


def test_explicit_sandbox_mcp_selection_is_preserved_exactly(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("FEDOTMAS_GAIA_MCP_SERVERS", "sandbox")
    monkeypatch.setenv("E2B_API_KEY", "test-key")

    assert _gaia_mcp_servers() == ["sandbox"]
    assert list(_gaia_mcp_registry(ModelConfig(model="openai/gpt-4o"))) == ["sandbox"]


def test_custom_task_file_preserves_raw_text_and_resolved_path(tmp_path: Path):
    task_file = tmp_path / "input task.txt"
    task_file.write_text("Complete raw task text.\nSecond line.", encoding="utf-8")

    task = _task_from_file(task_file)

    assert task.question == "Complete raw task text.\nSecond line."
    assert task.file_path == str(task_file.resolve())
    assert task.file_name == task_file.name


def test_startup_logs_meta_and_worker_models_independently():
    meta = ModelConfig(model="deepseek/meta")
    worker = ModelConfig(model="deepseek/worker")
    with patch("benchmarks.gaia.run_gaia._log.info") as log_info:
        _log_selected_models(meta, worker)
    log_info.assert_called_once_with(
        "Meta model:   {}\nWorker model: {}", "deepseek/meta", "deepseek/worker"
    )


def test_gaia_research_budget_defaults(monkeypatch: pytest.MonkeyPatch):
    for name in (
        "FEDOTMAS_GAIA_MAX_AGENT_LLM_TURNS",
        "FEDOTMAS_GAIA_WEB_SEARCH_LIMIT",
        "FEDOTMAS_GAIA_WEB_TOOL_LIMIT",
        "FEDOTMAS_GAIA_BROWSER_AGENT_LIMIT",
    ):
        monkeypatch.delenv(name, raising=False)

    plugins = build_plugins(SimpleNamespace(), enable_langfuse=False)
    budgets = {
        plugin.budget_kind: plugin.max_calls_per_agent
        for plugin in plugins
        if hasattr(plugin, "budget_kind")
    }
    assert _gaia_max_agent_llm_turns() == 20
    assert budgets["search"] == 40
    assert budgets["scraping"] == 40
    assert budgets["browser_agent"] == 3


@pytest.mark.asyncio
async def test_gaia_browser_limit_is_per_agent_and_uses_budget_control(monkeypatch):
    monkeypatch.setenv("FEDOTMAS_GAIA_BROWSER_AGENT_LIMIT", "3")
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
    assert summary["grand_total"] == {
        "prompt_tokens": 165,
        "completion_tokens": 30,
        "total_tokens": 195,
    }


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
    assert summary["grand_total"] == {
        "prompt_tokens": 130,
        "completion_tokens": 30,
        "total_tokens": 160,
    }
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
        initial_state: dict | None = None,
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
    assert result["pipeline_status"] == "completed"
    assert artifact["maw_config"] == _config().model_dump(mode="json")
    assert isinstance(artifact["research_telemetry"], dict)
    assert artifact["attempts"][0]["attempt_status"] == "succeeded"
    assert attempt["maw_config"] == _config().model_dump(mode="json")
    assert artifact["config_mode"] == "generated"


@pytest.mark.asyncio
async def test_valid_terminal_answer_survives_incomplete_intermediate_handoff(
    tmp_path: Path,
):
    class IncompleteIntermediateMAW(_FakeMAW):
        async def run(
            self,
            _query: str,
            *,
            timeout: int,
            final_answer_contract: str | None = None,
            initial_state: dict | None = None,
        ) -> dict[str, str]:
            self.last_result = PipelineResult(
                state={
                    "final_answer": "<solution>65</solution>",
                    "_fedotmas_execution": {
                        "handoff_issues": [
                            {
                                "kind": "incomplete_artifact",
                                "agent": "researcher",
                                "source_key": "findings",
                                "resolved": False,
                            }
                        ]
                    },
                },
                status="incomplete",
            )
            return self.last_result.state

    task = SimpleNamespace(
        task_id="task-valid-terminal-incomplete-handoff",
        question="Question?",
        ground_truth="65",
        file_path=None,
        file_name=None,
        difficulty="1",
    )

    with patch("benchmarks.gaia.run_gaia.MAW", IncompleteIntermediateMAW):
        result = await process_task(
            task,
            SimpleNamespace(is_correct_answer=lambda answer, truth: answer == truth),
            tmp_path,
            enable_langfuse=False,
        )

    attempt = json.loads((tmp_path / "attempts" / "attempt_01.json").read_text())
    saved_result = json.loads((tmp_path / "result.json").read_text())
    assert result["response"] == "65"
    assert result["pipeline_status"] == "incomplete"
    assert result["unresolved_execution_issues"][0]["kind"] == "incomplete_artifact"
    assert attempt["response"] == "65"
    assert attempt["pipeline_status"] == "incomplete"
    assert saved_result["response"] == "65"
    assert saved_result["pipeline_status"] == "incomplete"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("pipeline_status", "diagnostic_key", "diagnostic_value"),
    [
        (
            "incomplete",
            "_fedotmas_execution",
            {"contract_repairs": {"solver": [{"status": "repair_failed"}]}},
        ),
        (
            "limited",
            "_fedotmas_code_agent_budget",
            {"solver": {"status": "exhausted", "calls": 4}},
        ),
    ],
)
async def test_valid_terminal_answer_survives_intermediate_diagnostics(
    tmp_path: Path,
    pipeline_status: str,
    diagnostic_key: str,
    diagnostic_value: dict,
):
    class DiagnosticMAW(_FakeMAW):
        async def run(
            self,
            _query: str,
            *,
            timeout: int,
            final_answer_contract: str | None = None,
            initial_state: dict | None = None,
        ) -> dict[str, str]:
            self.last_result = PipelineResult(
                state={
                    "final_answer": "<solution>65</solution>",
                    diagnostic_key: diagnostic_value,
                },
                status=pipeline_status,
            )
            return self.last_result.state

    task = SimpleNamespace(
        task_id=f"task-terminal-with-{diagnostic_key}",
        question="Question?",
        ground_truth="65",
        file_path=None,
        file_name=None,
        difficulty="1",
    )
    with patch("benchmarks.gaia.run_gaia.MAW", DiagnosticMAW):
        result = await process_task(
            task,
            SimpleNamespace(is_correct_answer=lambda answer, truth: answer == truth),
            tmp_path,
            enable_langfuse=False,
        )

    assert result["response"] == "65"
    assert result["pipeline_status"] == pipeline_status
    assert result["pipeline_diagnostics"][diagnostic_key] == diagnostic_value


@pytest.mark.asyncio
async def test_explicit_terminal_abstention_remains_incomplete(tmp_path: Path):
    class AbstainingMAW(_FakeMAW):
        async def run(
            self,
            _query: str,
            *,
            timeout: int,
            final_answer_contract: str | None = None,
            initial_state: dict | None = None,
        ) -> dict[str, str]:
            self.last_result = PipelineResult(
                state={"final_answer": "<abstain>Unable to solve</abstain>"},
                status="incomplete",
            )
            return self.last_result.state

    task = SimpleNamespace(
        task_id="task-explicit-terminal-abstention",
        question="Question?",
        ground_truth="65",
        file_path=None,
        file_name=None,
        difficulty="1",
    )
    with (
        patch("benchmarks.gaia.run_gaia.MAW", AbstainingMAW),
        patch.dict("os.environ", {"FEDOTMAS_GAIA_TASK_ATTEMPTS": "1"}),
        pytest.raises(RuntimeError, match="explicitly abstained"),
    ):
        await process_task(
            task,
            SimpleNamespace(is_correct_answer=lambda answer, truth: answer == truth),
            tmp_path,
            enable_langfuse=False,
        )

    attempt = json.loads((tmp_path / "attempts" / "attempt_01.json").read_text())
    assert attempt["attempt_status"] == "incomplete"
    assert attempt["response"] == ""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "terminal",
    [
        "Upstream (recovered, preserved verbatim) is explicitly unresolved",
        '{"status":"UNRESOLVED","answer":null}',
        '{"verified_answer":null}',
    ],
)
async def test_explicit_terminal_non_answer_is_rejected(tmp_path: Path, terminal: str):
    class UnresolvedMAW(_FakeMAW):
        async def run(
            self,
            _query: str,
            *,
            timeout: int,
            final_answer_contract: str | None = None,
            initial_state: dict | None = None,
        ) -> dict[str, str]:
            self.last_result = PipelineResult(
                state={"final_answer": terminal}, status="incomplete"
            )
            return self.last_result.state

    task = SimpleNamespace(
        task_id="task-unresolved-terminal",
        question="Question?",
        ground_truth="65",
        file_path=None,
        file_name=None,
        difficulty="1",
        metadata={},
    )
    with (
        patch("benchmarks.gaia.run_gaia.MAW", UnresolvedMAW),
        patch.dict("os.environ", {"FEDOTMAS_GAIA_TASK_ATTEMPTS": "1"}),
        pytest.raises(RuntimeError, match="unresolved/non-answer"),
    ):
        await process_task(
            task,
            SimpleNamespace(is_correct_answer=lambda answer, truth: answer == truth),
            tmp_path,
            enable_langfuse=False,
        )
    attempt = json.loads((tmp_path / "attempts" / "attempt_01.json").read_text())
    assert attempt["attempt_status"] == "incomplete"
    assert attempt["response"] == ""


@pytest.mark.asyncio
async def test_incomplete_answer_lineage_is_not_accepted(tmp_path: Path):
    class BrokenLineageMAW(_FakeMAW):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.generated_config.agents[1].input_requirements = [
                ArtifactRequirement(source_key="solution", required_fields=["answer"])
            ]

        async def run(
            self,
            _query: str,
            *,
            timeout: int,
            final_answer_contract: str | None = None,
            initial_state: dict | None = None,
        ) -> dict[str, str]:
            self.last_result = PipelineResult(
                state={
                    "solution": '{"answer":null}',
                    "final_answer": "<solution>65</solution>",
                    "_fedotmas_execution": {
                        "handoff_issues": [
                            {
                                "kind": "incomplete_handoff",
                                "source_key": "solution",
                                "resolved": False,
                            }
                        ]
                    },
                },
                status="incomplete",
            )
            return self.last_result.state

    task = SimpleNamespace(
        task_id="task-broken-lineage",
        question="Question?",
        ground_truth="65",
        file_path=None,
        file_name=None,
        difficulty="1",
        metadata={},
    )
    with (
        patch("benchmarks.gaia.run_gaia.MAW", BrokenLineageMAW),
        patch.dict("os.environ", {"FEDOTMAS_GAIA_TASK_ATTEMPTS": "1"}),
        pytest.raises(RuntimeError, match="unresolved semantic answer lineage"),
    ):
        await process_task(
            task,
            SimpleNamespace(is_correct_answer=lambda answer, truth: answer == truth),
            tmp_path,
            enable_langfuse=False,
        )
    attempt = json.loads((tmp_path / "attempts" / "attempt_01.json").read_text())
    assert attempt["attempt_status"] == "incomplete"
    assert attempt["response"] == ""


@pytest.mark.asyncio
async def test_generated_and_frozen_configs_share_verifier_execution_policy(
    tmp_path: Path,
):
    payload = {
        "agents": [
            {
                "name": "solver",
                "instruction": "Solve.",
                "output_key": "solution",
                "tools": ["code-agent"],
                "model": "openai/gpt-4o",
            },
            {
                "name": "verifier",
                "instruction": "Verify the candidate.",
                "output_key": "verification",
                "tools": ["code-agent"],
                "model": "openai/gpt-4o",
                "input_requirements": [
                    {"source_key": "solution", "required_fields": ["answer"]}
                ],
            },
        ],
        "pipeline": {
            "type": "sequential",
            "children": [
                {"type": "agent", "agent_name": "solver"},
                {"type": "agent", "agent_name": "verifier"},
            ],
        },
    }
    generated = MAWConfig.model_validate(payload)
    frozen_path = tmp_path / "frozen.json"
    frozen_path.write_text(
        json.dumps({"maw_config": generated.model_dump(mode="json")})
    )
    frozen, _digest = _load_frozen_config(frozen_path)
    for config in (generated, frozen):
        tree = maw_builder.build(config, autonomous=False)
        state = {"solution": '{"answer":"candidate"}'}
        await tree.sub_agents[1].before_agent_callback(SimpleNamespace(state=state))
        assert (
            state[CODE_AGENT_POLICY_STATE_KEY]["verifier"]["mode"] == "verify_candidate"
        )


@pytest.mark.asyncio
async def test_terminal_agent_is_inferred_when_config_omits_optional_field(
    tmp_path: Path,
):
    class OmittedTerminalMAW(_FakeMAW):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.generated_config = _config().model_copy(
                update={"final_answer_agent": None}
            )

    task = SimpleNamespace(
        task_id="task-inferred-terminal",
        question="Question?",
        ground_truth="42",
        file_path="",
        file_name="",
        difficulty="1",
    )
    with patch("benchmarks.gaia.run_gaia.MAW", OmittedTerminalMAW):
        result = await process_task(
            task,
            SimpleNamespace(is_correct_answer=lambda answer, truth: answer == truth),
            tmp_path,
            enable_langfuse=False,
        )
    assert result["response"] == "42"
    assert result["is_correct"] is True
    assert result["pipeline_status"] == "completed"


@pytest.mark.asyncio
async def test_custom_task_has_no_score_and_is_excluded_from_accuracy(
    tmp_path: Path,
):
    task = SimpleNamespace(
        task_id="custom_exact_optimization",
        question="Raw task",
        ground_truth="",
        file_path="",
        file_name="",
        difficulty="0",
        metadata={"source": "custom_task_file"},
    )
    with patch("benchmarks.gaia.run_gaia.MAW", _FakeMAW):
        result = await process_task(
            task,
            SimpleNamespace(
                is_correct_answer=lambda *_: pytest.fail("custom tasks are unscored")
            ),
            tmp_path,
            enable_langfuse=False,
        )
    assert result["is_correct"] is None
    assert compute_metrics_by_level([result])["overall"] == {
        "total_tasks": 0,
        "correct": 0,
        "accuracy": 0,
    }


def test_empty_scored_summary_reports_na(capsys):
    print_score_by_level(
        compute_metrics_by_level([{"difficulty": "0", "is_correct": None}])
    )
    output = capsys.readouterr().out
    assert "Overall:   N/A (no scored tasks)" in output
    assert "0.00%  (0/0)" not in output


def test_declared_answer_format_is_checked_for_incomplete_results():
    numeric_task = SimpleNamespace(metadata={"answer_format": "integer"})
    assert _matches_declared_answer_format("42", numeric_task)
    assert not _matches_declared_answer_format("forty two", numeric_task)


def test_nested_terminal_answer_fields_keep_unresolved_lineage_blocking():
    terminal = SimpleNamespace(
        input_requirements=[
            SimpleNamespace(
                source_key="upstream",
                required_fields=["result.answer"],
                identity_fields=[],
            )
        ]
    )
    assert _has_unresolved_answer_lineage(
        [{"kind": "incomplete_handoff", "source_key": "upstream"}], terminal
    )


@pytest.mark.asyncio
async def test_postprocessing_failure_does_not_rerun_completed_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    class CountedMAW(_FakeMAW):
        calls = 0

        async def run(self, *args, **kwargs):
            type(self).calls += 1
            return await super().run(*args, **kwargs)

    monkeypatch.setenv("FEDOTMAS_GAIA_TASK_ATTEMPTS", "3")
    task = SimpleNamespace(
        task_id="task-postprocessing",
        question="Question?",
        ground_truth="42",
        file_path="",
        file_name="",
        difficulty="1",
    )
    with patch("benchmarks.gaia.run_gaia.MAW", CountedMAW):
        result = await process_task(
            task,
            SimpleNamespace(
                is_correct_answer=lambda *_: (_ for _ in ()).throw(
                    ValueError("scoring failed")
                )
            ),
            tmp_path,
            enable_langfuse=False,
        )
    assert CountedMAW.calls == 1
    assert result["response"] == "42"
    assert result["error"] == "scoring failed"
    assert result["is_correct"] is False


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
            initial_state: dict | None = None,
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
    assert attempt["config_mode"] == "generated"
    assert isinstance(artifact["research_telemetry"], dict)


def test_frozen_config_loads_raw_config_and_hash_is_canonical(tmp_path: Path):
    compact = tmp_path / "compact.json"
    formatted = tmp_path / "formatted.json"
    payload = _config().model_dump(mode="json")
    compact.write_text(json.dumps(payload, separators=(",", ":")))
    formatted.write_text(json.dumps(payload, indent=4))

    config_a, hash_a = _load_frozen_config(compact)
    config_b, hash_b = _load_frozen_config(formatted)

    assert config_a == _config()
    assert config_b == config_a
    assert hash_a == hash_b


def test_frozen_config_loads_gaia_result_artifact(tmp_path: Path):
    artifact = tmp_path / "result.json"
    artifact.write_text(json.dumps({"maw_config": _config().model_dump(mode="json")}))

    config, digest = _load_frozen_config(artifact)

    assert config == _config()
    assert len(digest) == 64


@pytest.mark.parametrize(
    ("contents", "error"),
    [
        ("{", "Invalid JSON"),
        ('{"maw_config": null}', "missing or null"),
        ('{"task_id": "task-1"}', "missing maw_config"),
        ('{"agents": []}', "Invalid MAWConfig"),
    ],
)
def test_frozen_config_invalid_inputs_fail_clearly(
    tmp_path: Path, contents: str, error: str
):
    path = tmp_path / "bad.json"
    path.write_text(contents)
    with pytest.raises(ValueError, match=error):
        _load_frozen_config(path)


def test_frozen_config_missing_file_fails_clearly(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Frozen MAW config file not found"):
        _load_frozen_config(tmp_path / "absent.json")


@pytest.mark.asyncio
async def test_invalid_frozen_config_fails_before_maw_generation(tmp_path: Path):
    from benchmarks.gaia.run_gaia import run_gaia

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{")
    with (
        patch("benchmarks.gaia.run_gaia.MAW") as maw_class,
        pytest.raises(ValueError, match="Invalid JSON"),
    ):
        await run_gaia(
            "all",
            "validation",
            enable_langfuse=False,
            task_file="task.txt",
            frozen_config_path=str(invalid),
        )
    maw_class.assert_not_called()


@pytest.mark.asyncio
async def test_frozen_execution_builds_supplied_config_without_generation(
    tmp_path: Path,
):
    class FrozenMAW(_FakeMAW):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.generated_config = None
            self.meta_prompt_tokens = 0
            self.meta_completion_tokens = 0
            self.elapsed = 1.5
            self.run_called = False
            self.build_args = None

        async def run(self, *args, **kwargs):
            self.run_called = True
            raise AssertionError("frozen execution must not call run")

        async def build_and_run(self, config, query, **kwargs):
            self.generated_config = config
            self.build_args = (query, kwargs)
            return {"final_answer": "Reasoning <solution>42</solution>"}

    task = SimpleNamespace(
        task_id="frozen-task",
        question="Question?",
        ground_truth="42",
        file_path=None,
        file_name=None,
        difficulty="1",
    )
    frozen = _config()
    digest = "a" * 64
    with patch("benchmarks.gaia.run_gaia.MAW", FrozenMAW):
        result = await process_task(
            task,
            SimpleNamespace(is_correct_answer=lambda answer, truth: answer == truth),
            tmp_path,
            enable_langfuse=False,
            frozen_config=frozen,
            frozen_config_source="frozen.json",
            frozen_config_sha256=digest,
        )

    artifact = json.loads((tmp_path / "result.json").read_text())
    attempt = json.loads((tmp_path / "attempts" / "attempt_01.json").read_text())
    assert result["response"] == "42"
    assert result["is_correct"] is True
    assert result["tokens"]["meta_prompt"] == 0
    assert result["tokens"]["meta_completion"] == 0
    for saved in (artifact, attempt):
        assert saved["config_mode"] == "frozen"
        assert saved["frozen_config_source"] == "frozen.json"
        assert saved["frozen_config_sha256"] == digest
        assert saved["maw_config"] == frozen.model_dump(mode="json")


@pytest.mark.asyncio
async def test_frozen_execution_failure_persists_config_provenance(tmp_path: Path):
    class FailingFrozenMAW(_FakeMAW):
        async def run(self, *args, **kwargs):
            raise AssertionError("frozen execution must not call run")

        async def build_and_run(self, config, query, **kwargs):
            self.generated_config = config
            raise RuntimeError("frozen execution failed")

    task = SimpleNamespace(
        task_id="frozen-failed-task",
        question="Question?",
        ground_truth="42",
        file_path=None,
        file_name=None,
        difficulty="1",
    )
    with (
        patch("benchmarks.gaia.run_gaia.MAW", FailingFrozenMAW),
        patch.dict("os.environ", {"FEDOTMAS_GAIA_TASK_ATTEMPTS": "1"}),
        pytest.raises(RuntimeError, match="frozen execution failed"),
    ):
        await process_task(
            task,
            SimpleNamespace(),
            tmp_path,
            enable_langfuse=False,
            frozen_config=_config(),
            frozen_config_source="frozen.json",
            frozen_config_sha256="b" * 64,
        )

    artifact = json.loads((tmp_path / "result.json").read_text())
    attempt = json.loads((tmp_path / "attempts" / "attempt_01.json").read_text())
    for saved in (artifact, attempt):
        assert saved["config_mode"] == "frozen"
        assert saved["frozen_config_source"] == "frozen.json"
        assert saved["frozen_config_sha256"] == "b" * 64
        assert saved["maw_config"] == _config().model_dump(mode="json")


def test_frozen_config_cli_requires_task_file(monkeypatch: pytest.MonkeyPatch):
    from benchmarks.gaia.run_gaia import main

    monkeypatch.setattr(sys, "argv", ["run_gaia.py", "--frozen-config", "config.json"])
    with pytest.raises(SystemExit, match="2"):
        main()


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
            initial_state: dict | None = None,
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
            initial_state: dict | None = None,
        ) -> dict[str, str]:
            self.last_result = PipelineResult(
                state={
                    "research_findings": partial_answer,
                    "_fedotmas_execution": {
                        "handoff_issues": [
                            {
                                "kind": "incomplete_handoff",
                                "agent": "researcher",
                                "source_key": "findings",
                                "resolved": False,
                            }
                        ]
                    },
                },
                status="timed_out",
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
    assert attempt["unresolved_execution_issues"] == [
        {
            "kind": "incomplete_handoff",
            "agent": "researcher",
            "source_key": "findings",
            "resolved": False,
        }
    ]
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
            initial_state: dict | None = None,
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
