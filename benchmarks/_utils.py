from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fedotmas.optimize._scoring import ScoringResult
from fedotmas.optimize._state import Task
from pydantic import BaseModel


class CostSummary(BaseModel):
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


class TaskResult(BaseModel):
    task_id: str
    input: str
    expected: str | None = None
    output: str
    score: float
    correct: bool
    metadata: dict[str, Any] = {}


class BenchmarkResult(BaseModel):
    benchmark: str
    metrics: dict[str, float]
    cost: CostSummary | None = None
    iterations: int | None = None
    per_task: list[TaskResult] = []


def save_result(result: BenchmarkResult, output_dir: Path | str) -> Path:
    """Persist *result* as a timestamped JSON file under *output_dir*."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    path = output_dir / f"{result.benchmark}_{ts}.json"
    path.write_text(result.model_dump_json(indent=2))
    return path


def load_result(path: Path | str) -> BenchmarkResult:
    """Load a previously saved benchmark result."""
    return BenchmarkResult.model_validate_json(Path(path).read_text())


class ExactMatchScorer:
    def __init__(
        self,
        output_key: str = "answer",
        normalize_fn: Callable[[str], str] | None = None,
    ) -> None:
        self._output_key = output_key
        self._normalize = normalize_fn or (lambda s: s.strip())

    async def evaluate(self, task: Task, state: dict[str, Any]) -> ScoringResult:
        raw = str(state.get(self._output_key, ""))
        try:
            predicted = self._normalize(raw)
        except Exception as exc:
            return ScoringResult(
                score=0.0,
                feedback=f"Normalization failed on {raw!r}: {exc}",
                reasoning="normalize_fn raised an exception",
            )

        expected = task.expected or ""
        correct = predicted == expected
        return ScoringResult(
            score=1.0 if correct else 0.0,
            feedback=f"Expected {expected!r}, got {predicted!r}.",
            reasoning=f"Exact match: {correct}",
        )
