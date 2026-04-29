from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any

from fedotmas.optimize._scoring import ScoringResult
from fedotmas.optimize._state import Task


def normalize(s: str) -> str:
    """SQuAD-style normalization: lowercase, strip articles/punct/extra ws."""
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())


def _f1(pred: str, gold: str) -> float:
    pred_tokens = normalize(pred).split()
    gold_tokens = normalize(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(pred: str, gold: str) -> bool:
    return normalize(pred) == normalize(gold)


class HotpotQAScorer:
    """SQuAD-style F1 scorer for short-answer QA.

    The continuous F1 score (0-1) is what the optimizer accepts/rejects on,
    so partial overlaps still produce a usable gradient. Exact-match is
    surfaced separately via the feedback string but does not feed scoring.
    """

    def __init__(self, output_key: str = "answer") -> None:
        self._output_key = output_key

    async def evaluate(self, task: Task, state: dict[str, Any]) -> ScoringResult:
        raw = str(state.get(self._output_key, "")).strip()
        expected = task.expected or ""

        if not raw:
            return ScoringResult(
                score=0.0,
                feedback=(
                    f"No answer produced. The correct answer is {expected!r}."
                ),
                reasoning="Empty output",
            )

        f1 = _f1(raw, expected)
        em = exact_match(raw, expected)

        if em:
            return ScoringResult(
                score=1.0,
                feedback=f"Your answer is correct. The correct answer is {expected!r}.",
                reasoning=f"Exact match: {raw!r} ≈ {expected!r}",
            )

        feedback = (
            f"Your answer {raw[:200]!r} is not an exact match. "
            f"The correct answer is {expected!r}. "
            f"Token-level F1 overlap: {f1:.2f}."
        )
        return ScoringResult(
            score=f1,
            feedback=feedback,
            reasoning=f"F1 = {f1:.3f} between {raw[:200]!r} and {expected!r}",
        )
