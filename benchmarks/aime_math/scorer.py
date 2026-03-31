from __future__ import annotations

from typing import Any

from fedotmas.optimize._scoring import ScoringResult
from fedotmas.optimize._state import Task


class ExactIntScorer:
    """Scores a math-solver agent by exact integer comparison.

    Reads the agent answer from ``state[output_key]``, parses it as ``int``,
    and compares against ``int(task.expected)``.
    """

    def __init__(self, output_key: str = "answer") -> None:
        self._output_key = output_key

    async def evaluate(self, task: Task, state: dict[str, Any]) -> ScoringResult:
        raw = str(state.get(self._output_key, ""))

        try:
            predicted = int(raw.strip())
        except (ValueError, TypeError):
            return ScoringResult(
                score=0.0,
                feedback=(
                    f"Could not parse integer from {raw!r}. "
                    f"The correct answer is {task.expected!r}."
                ),
                reasoning="Failed to parse agent output as int",
            )

        expected = int(task.expected)  # type: ignore[arg-type]
        correct = predicted == expected
        status = "correct" if correct else "incorrect"
        return ScoringResult(
            score=1.0 if correct else 0.0,
            feedback=f"Your answer is {status}. The correct answer is {expected!r}.",
            reasoning=f"Exact int match: {predicted} == {expected} → {correct}",
        )
