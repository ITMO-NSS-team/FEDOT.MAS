"""Private evaluator for public MAS predictions; never exposed to agents."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from sampo_evaluation import evaluate_predictions

ROOT = Path(__file__).resolve().parents[1]
GT = ROOT / "artifacts" / "sampo_audit" / "private_ground_truth.csv"
PRED = ROOT / "artifacts" / "sampo_benchmark" / "mas_runs" / "full_predictions.csv"
OUT = ROOT / "artifacts" / "sampo_mas_experiment" / "private_evaluation.json"


def main() -> None:
    ground_truth = list(csv.DictReader(GT.open(encoding="utf-8", newline="")))
    for index, row in enumerate(ground_truth, start=1):
        row["example_id"] = str(index)
    labels = sorted({row["target_granular_name"] for row in ground_truth})
    predictions = list(csv.DictReader(PRED.open(encoding="utf-8", newline="")))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            evaluate_predictions(ground_truth, predictions, labels), indent=2
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
