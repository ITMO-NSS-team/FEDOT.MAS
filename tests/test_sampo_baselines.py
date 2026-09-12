from __future__ import annotations

import sys
from itertools import pairwise
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from sampo_baselines import tfidf_char_ngrams, tfidf_char_ngrams_ranked


def test_char_ranked_returns_scores_for_arbitrary_k() -> None:
    labels = ["alpha", "beta", "gamma", "delta"]
    ranking = tfidf_char_ngrams_ranked(["alpha beta"], labels, 4)[0]

    assert len(ranking) == 4
    assert {label for label, _ in ranking} == set(labels)
    assert all(first[1] >= second[1] for first, second in pairwise(ranking))


def test_char_baseline_keeps_its_label_only_interface() -> None:
    labels = ["alpha", "beta", "gamma", "delta"]
    ranking = tfidf_char_ngrams_ranked(["alpha beta"], labels, 3)[0]
    expected = [label for label, score in ranking if score > 0]
    expected.extend([None] * (3 - len(expected)))

    assert tfidf_char_ngrams(["alpha beta"], labels, 3) == [expected]
