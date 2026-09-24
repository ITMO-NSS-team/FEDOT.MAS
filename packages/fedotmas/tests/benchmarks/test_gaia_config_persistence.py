from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fedotmas.maw.models import MAWConfig

from benchmarks.gaia.run_gaia import process_task


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

    assert result["is_correct"] is True
    assert artifact["maw_config"] == _config().model_dump(mode="json")
