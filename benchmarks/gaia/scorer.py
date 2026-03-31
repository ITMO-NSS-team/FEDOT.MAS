from __future__ import annotations

import re
import string
from typing import Any

from fedotmas.optimize._scoring import ScoringResult
from fedotmas.optimize._state import Task

_SOLUTION_RE = re.compile(r"<solution>(.*?)</solution>", re.DOTALL)


def _extract_solution(text: str) -> str:
    matches = _SOLUTION_RE.findall(text)
    return matches[-1].strip() if matches else text.strip()


def _normalize_number(s: str) -> float:
    for ch in ("$", "%", ","):
        s = s.replace(ch, "")
    try:
        return float(s)
    except ValueError:
        return float("inf")


def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def _normalize_str(s: str, remove_punct: bool = True) -> str:
    no_spaces = re.sub(r"\s", "", s)
    if remove_punct:
        translator = str.maketrans("", "", string.punctuation)
        return no_spaces.lower().translate(translator)
    return no_spaces.lower()


def _split_string(s: str) -> list[str]:
    return re.split(r"[,;]", s)


def is_correct_answer(model_answer: str, ground_truth: str) -> bool:
    """GAIA answer comparison with number/list/string normalization."""
    if _is_float(ground_truth):
        return _normalize_number(model_answer) == float(ground_truth)

    if any(ch in ground_truth for ch in (",", ";")):
        gt_elems = _split_string(ground_truth)
        ma_elems = _split_string(model_answer)
        if len(gt_elems) != len(ma_elems):
            return False
        return all(
            _normalize_number(ma) == float(gt)
            if _is_float(gt)
            else _normalize_str(ma, remove_punct=False)
            == _normalize_str(gt, remove_punct=False)
            for ma, gt in zip(ma_elems, gt_elems)
        )

    return _normalize_str(model_answer) == _normalize_str(ground_truth)


class GaiaScorer:
    """Scorer for GAIA benchmark using official normalization rules."""

    def __init__(self, output_key: str = "answer") -> None:
        self._output_key = output_key

    async def evaluate(self, task: Task, state: dict[str, Any]) -> ScoringResult:
        raw = str(state.get(self._output_key, ""))
        predicted = _extract_solution(raw)
        expected = task.expected or ""

        correct = is_correct_answer(predicted, expected)
        return ScoringResult(
            score=1.0 if correct else 0.0,
            feedback=f"Expected {expected!r}, got {predicted!r}.",
            reasoning=f"GAIA match: {correct}",
        )
