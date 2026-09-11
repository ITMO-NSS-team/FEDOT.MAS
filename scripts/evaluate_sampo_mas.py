"""Private evaluator for public MAS predictions; never exposed to agents."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GT = ROOT / "artifacts" / "sampo_audit" / "private_ground_truth.csv"
PRED = ROOT / "artifacts" / "sampo_benchmark" / "mas_runs" / "full_predictions.csv"
OUT = ROOT / "artifacts" / "sampo_mas_experiment" / "private_evaluation.json"


def metrics(
    targets: list[str], predictions: list[list[str]], labels: list[str]
) -> dict[str, float]:
    top_1 = sum(row[0] == target for target, row in zip(targets, predictions)) / len(
        targets
    )
    top_3 = sum(target in row[:3] for target, row in zip(targets, predictions)) / len(
        targets
    )
    f1_values = []
    for label in labels:
        true_positive = sum(
            target == label and row[0] == label
            for target, row in zip(targets, predictions)
        )
        false_positive = sum(
            target != label and row[0] == label
            for target, row in zip(targets, predictions)
        )
        false_negative = sum(
            target == label and row[0] != label
            for target, row in zip(targets, predictions)
        )
        denominator = 2 * true_positive + false_positive + false_negative
        f1_values.append(2 * true_positive / denominator if denominator else 0.0)
    return {
        "top_1_accuracy": top_1,
        "top_3_accuracy": top_3,
        "macro_f1": sum(f1_values) / len(f1_values),
    }


targets = [
    row["target_granular_name"] for row in csv.DictReader(GT.open(encoding="utf-8"))
]
labels = sorted(set(targets))
rows = list(csv.DictReader(PRED.open(encoding="utf-8")))
if len(rows) != len(targets):
    raise RuntimeError(
        f"Prediction count {len(rows)} does not match private GT {len(targets)}"
    )
predictions = [[row["top_1"], row["top_2"], row["top_3"]] for row in rows]
OUT.write_text(
    json.dumps(metrics(targets, predictions, labels), indent=2) + "\n", encoding="utf-8"
)
