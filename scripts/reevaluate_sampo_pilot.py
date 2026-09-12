"""Privately re-evaluate the existing semantic pilot and record comparability."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from sampo_evaluation import PredictionValidationError, evaluate_predictions

ROOT = Path(__file__).resolve().parents[1]
GT = ROOT / "artifacts" / "sampo_audit" / "private_ground_truth.csv"
PREDICTIONS = (
    ROOT / "artifacts" / "sampo_benchmark" / "mas_runs" / "semantic_pilot_predictions.csv"
)
MANIFEST = ROOT / "artifacts" / "sampo_benchmark" / "pilot_manifest.json"
OUT = ROOT / "artifacts" / "sampo_mas_semantic_experiment" / "existing_pilot_evaluation.json"


def main() -> None:
    ground_truth = list(csv.DictReader(GT.open(encoding="utf-8", newline="")))
    for index, row in enumerate(ground_truth, start=1):
        row["example_id"] = str(index)
    labels = sorted({row["target_granular_name"] for row in ground_truth})
    predictions = list(csv.DictReader(PREDICTIONS.open(encoding="utf-8", newline="")))
    manifest_ids = set(json.loads(MANIFEST.read_text(encoding="utf-8"))["example_ids"])
    prediction_ids = {row.get("example_id", "") for row in predictions}
    result: dict[str, object] = {
        "prediction_path": str(PREDICTIONS.relative_to(ROOT)),
        "comparison_pilot_manifest": str(MANIFEST.relative_to(ROOT)),
        "comparable_to_seed_42_pilot": prediction_ids == manifest_ids,
    }
    if prediction_ids != manifest_ids:
        result["status"] = "non_comparable"
        result["reason"] = "Existing pilot IDs differ from the fixed seed-42 pilot IDs."
    else:
        result["status"] = "comparable"
    try:
        result["metrics"] = evaluate_predictions(
            [row for row in ground_truth if row["example_id"] in prediction_ids],
            predictions,
            labels,
        )
    except PredictionValidationError as error:
        result["metrics_status"] = "not_evaluated"
        result["evaluation_error"] = str(error)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
