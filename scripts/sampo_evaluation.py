"""Private, ID-based evaluation helpers for SAMPO predictions."""

from __future__ import annotations

from collections.abc import Iterable

PREDICTION_COLUMNS = ("top_1", "top_2", "top_3")


class PredictionValidationError(ValueError):
    """Raised when a prediction file cannot be evaluated safely."""


def validate_and_join_predictions(
    ground_truth: Iterable[dict[str, str]],
    predictions: Iterable[dict[str, str]],
    allowed_labels: Iterable[str],
) -> list[tuple[dict[str, str], dict[str, str]]]:
    """Validate prediction rows and join them to GT by ``example_id``."""
    ground_truth_by_id: dict[str, dict[str, str]] = {}
    for row in ground_truth:
        example_id = row.get("example_id", "")
        if not example_id:
            raise PredictionValidationError("Ground truth contains an empty example_id")
        if example_id in ground_truth_by_id:
            raise PredictionValidationError(f"Ground truth contains duplicate ID: {example_id}")
        ground_truth_by_id[example_id] = row

    allowed = set(allowed_labels)
    predictions_by_id: dict[str, dict[str, str]] = {}
    for row in predictions:
        example_id = row.get("example_id", "")
        if not example_id:
            raise PredictionValidationError("Prediction contains an empty example_id")
        if example_id in predictions_by_id:
            raise PredictionValidationError(f"Prediction contains duplicate ID: {example_id}")
        if example_id not in ground_truth_by_id:
            raise PredictionValidationError(f"Prediction contains unknown ID: {example_id}")
        labels = [row.get(column, "") for column in PREDICTION_COLUMNS]
        unknown = [label for label in labels if label not in allowed]
        if unknown:
            raise PredictionValidationError(
                f"Prediction for ID {example_id} contains unknown label(s): {unknown}"
            )
        if len(set(labels)) != len(labels):
            raise PredictionValidationError(
                f"Prediction for ID {example_id} has duplicate top-1/top-2/top-3 labels"
            )
        predictions_by_id[example_id] = row

    missing = set(ground_truth_by_id) - set(predictions_by_id)
    if missing:
        raise PredictionValidationError(
            f"Predictions are missing expected ID(s): {', '.join(sorted(missing))}"
        )
    extra = set(predictions_by_id) - set(ground_truth_by_id)
    if extra:
        raise PredictionValidationError(
            f"Predictions contain extra ID(s): {', '.join(sorted(extra))}"
        )
    return [
        (ground_truth_by_id[example_id], predictions_by_id[example_id])
        for example_id in ground_truth_by_id
    ]


def evaluate_predictions(
    ground_truth: Iterable[dict[str, str]],
    predictions: Iterable[dict[str, str]],
    allowed_labels: Iterable[str],
) -> dict[str, float | int]:
    """Return ID-aligned metrics, including observed-label and all-label F1."""
    allowed = list(allowed_labels)
    joined = validate_and_join_predictions(ground_truth, predictions, allowed)
    if not joined:
        raise PredictionValidationError("Cannot evaluate an empty prediction set")

    targets = [truth["target_granular_name"] for truth, _ in joined]
    top_predictions = [prediction["top_1"] for _, prediction in joined]
    observed_labels = sorted(set(targets))

    def macro_f1(labels: Iterable[str]) -> float:
        values = []
        for label in labels:
            true_positive = sum(
                target == label and prediction == label
                for target, prediction in zip(targets, top_predictions)
            )
            false_positive = sum(
                target != label and prediction == label
                for target, prediction in zip(targets, top_predictions)
            )
            false_negative = sum(
                target == label and prediction != label
                for target, prediction in zip(targets, top_predictions)
            )
            denominator = 2 * true_positive + false_positive + false_negative
            values.append(2 * true_positive / denominator if denominator else 0.0)
        return sum(values) / len(values) if values else 0.0

    return {
        "examples": len(joined),
        "top_1_accuracy": sum(
            target == prediction for target, prediction in zip(targets, top_predictions)
        )
        / len(joined),
        "top_3_accuracy": sum(
            truth["target_granular_name"]
            in [prediction[column] for column in PREDICTION_COLUMNS]
            for truth, prediction in joined
        )
        / len(joined),
        "macro_f1_observed_labels": macro_f1(observed_labels),
        "macro_f1_all_allowed_labels": macro_f1(allowed),
        "observed_label_count": len(observed_labels),
        "allowed_label_count": len(allowed),
        "label_coverage": len(observed_labels) / len(allowed) if allowed else 0.0,
    }
