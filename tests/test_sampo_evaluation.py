from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from sampo_evaluation import PredictionValidationError, evaluate_predictions


LABELS = ["a", "b", "c"]
GROUND_TRUTH = [
    {"example_id": "1", "target_granular_name": "a"},
    {"example_id": "2", "target_granular_name": "b"},
]
PREDICTIONS = [
    {"example_id": "1", "top_1": "a", "top_2": "b", "top_3": "c"},
    {"example_id": "2", "top_1": "c", "top_2": "b", "top_3": "a"},
]


def test_shuffled_prediction_rows_have_identical_metrics() -> None:
    assert evaluate_predictions(GROUND_TRUTH, PREDICTIONS, LABELS) == evaluate_predictions(
        GROUND_TRUTH, list(reversed(PREDICTIONS)), LABELS
    )


def test_wrong_id_fails() -> None:
    rows = [*PREDICTIONS]
    rows[1] = {**rows[1], "example_id": "999"}
    with pytest.raises(PredictionValidationError, match="unknown ID"):
        evaluate_predictions(GROUND_TRUTH, rows, LABELS)


def test_missing_id_fails() -> None:
    with pytest.raises(PredictionValidationError, match="missing expected ID"):
        evaluate_predictions(GROUND_TRUTH, PREDICTIONS[:1], LABELS)


def test_duplicate_id_fails() -> None:
    rows = [PREDICTIONS[0], {**PREDICTIONS[1], "example_id": "1"}]
    with pytest.raises(PredictionValidationError, match="duplicate ID"):
        evaluate_predictions(GROUND_TRUTH, rows, LABELS)


def test_unknown_and_duplicate_labels_fail() -> None:
    unknown = [{**PREDICTIONS[0], "top_3": "unknown"}, PREDICTIONS[1]]
    with pytest.raises(PredictionValidationError, match="unknown label"):
        evaluate_predictions(GROUND_TRUTH, unknown, LABELS)
    duplicated = [{**PREDICTIONS[0], "top_3": "a"}, PREDICTIONS[1]]
    with pytest.raises(PredictionValidationError, match="duplicate top-1/top-2/top-3"):
        evaluate_predictions(GROUND_TRUTH, duplicated, LABELS)
