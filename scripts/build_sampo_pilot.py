"""Build public artifacts for the fixed deterministic SAMPO pilot."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

from sampo_baselines import tfidf_char_ngrams

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "artifacts" / "sampo_benchmark"
PILOT_SEED = 42
PILOT_SIZE = 1000


def main() -> None:
    with (PUBLIC / "benchmark_inputs.csv").open(encoding="utf-8", newline="") as file:
        inputs = list(csv.DictReader(file))
    with (PUBLIC / "allowed_target_labels.csv").open(
        encoding="utf-8", newline=""
    ) as file:
        labels = [row["target_label"] for row in csv.DictReader(file)]
    pilot_rows = random.Random(PILOT_SEED).sample(inputs, PILOT_SIZE)
    with (PUBLIC / "pilot_inputs.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["example_id", "raw_work_name"])
        writer.writeheader()
        writer.writerows(pilot_rows)
    (PUBLIC / "pilot_manifest.json").write_text(
        json.dumps(
            {
                "selection": "random.sample without replacement",
                "seed": PILOT_SEED,
                "sample_size": PILOT_SIZE,
                "source": "benchmark_inputs.csv",
                "example_ids": [row["example_id"] for row in pilot_rows],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    predictions = tfidf_char_ngrams(
        [row["raw_work_name"] for row in pilot_rows], labels, 3
    )
    with (PUBLIC / "predictions_tfidf_char_ngrams_pilot.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=["example_id", "top_1", "top_2", "top_3"]
        )
        writer.writeheader()
        writer.writerows(
            {
                "example_id": row["example_id"],
                "top_1": prediction[0],
                "top_2": prediction[1],
                "top_3": prediction[2],
            }
            for row, prediction in zip(pilot_rows, predictions)
        )


if __name__ == "__main__":
    main()
