from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fedotmas.core.runner import PipelineExecutionError, PipelineResult
from fedotmas.maw.models import MAWConfig

from benchmarks.gaia.run_gaia import (
    compute_token_summary,
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
            "pipeline": {
                "type": "sequential",
                "children": [
                    {"type": "agent", "agent_name": "researcher"},
                    {"type": "agent", "agent_name": "answerer"},
                ],
            },
        }
    )


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

    async def run(self, _query: str, *, timeout: int) -> dict[str, str]:
        assert timeout > 0
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

        async def run(self, _query: str, *, timeout: int) -> dict[str, str]:
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

        async def run(self, _query: str, *, timeout: int) -> dict[str, str]:
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
