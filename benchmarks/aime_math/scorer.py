from __future__ import annotations

import re
from typing import Any

from fedotmas.optimize._scoring import ScoringResult
from fedotmas.optimize._state import Task

# Patterns to extract a boxed answer like \boxed{123} (LaTeX convention),
# or a final "answer is 123" / "= 123" line.
_BOXED_RE = re.compile(r"\\boxed\{(\d+)\}")
_ANSWER_IS_RE = re.compile(r"(?:answer|result)\s*(?:is|=|:)\s*(\d+)", re.IGNORECASE)
_TRAILING_NUM_RE = re.compile(r"(\d+)\s*$")


def _extract_int(text: str) -> int | None:
    """Try to extract the final integer answer from free-form text."""
    text = text.strip()
    # 1) Direct int parse
    try:
        return int(text)
    except (ValueError, TypeError):
        pass
    # 2) \boxed{N}
    m = _BOXED_RE.search(text)
    if m:
        return int(m.group(1))
    # 3) "answer is N" / "= N"
    matches = _ANSWER_IS_RE.findall(text)
    if matches:
        return int(matches[-1])
    # 4) Last number in text
    m = _TRAILING_NUM_RE.search(text)
    if m:
        return int(m.group(1))
    return None


class ExactIntScorer:
    """Scores a math-solver agent by exact integer comparison.

    Reads the agent answer from ``state[output_key]``, extracts the final
    integer (supports \\boxed{}, "answer is N", or trailing number),
    and compares against ``int(task.expected)``. If a reference solution
    is available, it is appended to the feedback (GEPA-style) so that
    the reflector can learn from worked examples.
    """

    def __init__(
        self,
        output_key: str = "answer",
        solutions: dict[str, str] | None = None,
    ) -> None:
        self._output_key = output_key
        self._solutions = solutions or {}

    def _solution_suffix(self, task: Task) -> str:
        sol = self._solutions.get(task.input)
        if not sol:
            return ""
        return (
            f" Here's the full step-by-step solution:\n{sol}\n\n"
            "Think about what takeaways you can learn from this solution "
            "to improve your future answers and approach to similar problems"
        )

    async def evaluate(self, task: Task, state: dict[str, Any]) -> ScoringResult:
        raw = str(state.get(self._output_key, ""))
        predicted = _extract_int(raw)
        suffix = self._solution_suffix(task)

        if predicted is None:
            return ScoringResult(
                score=0.0,
                feedback=(
                    f"The final answer must be a valid integer and nothing else. "
                    f"You responded with {raw[:200]!r}, which couldn't be parsed "
                    f"as an integer. The correct answer is {task.expected!r}.{suffix}"
                ),
                reasoning=f"Failed to extract int from: {raw[:200]!r}",
            )

        expected = int(task.expected)  # type: ignore[arg-type]
        correct = predicted == expected
        status = "correct" if correct else "incorrect"
        return ScoringResult(
            score=1.0 if correct else 0.0,
            feedback=f"Your answer is {status}. The correct answer is {expected!r}.{suffix}",
            reasoning=f"Exact int match: {predicted} == {expected} → {correct}",
        )
