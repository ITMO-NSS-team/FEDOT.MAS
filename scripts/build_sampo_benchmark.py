"""Build and evaluate the private SAMPO entity-resolution benchmark."""

from __future__ import annotations

import csv
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import psycopg
from sampo_baselines import BASELINES
from sampo_evaluation import evaluate_predictions

ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = ROOT / "artifacts" / "sampo_audit"
OUTPUT = ROOT / "artifacts" / "sampo_benchmark"
GT_PATH = AUDIT_DIR / "private_ground_truth.csv"
PILOT_SEED = 42
PILOT_SIZE = 1000


def dsn() -> str:
    return (
        os.getenv("SAMPO_BENCHMARK_DSN")
        or os.getenv("SAMPO_AUDIT_DSN")
        or "dbname=sampo"
    )


def read_ground_truth() -> list[dict[str, str]]:
    with GT_PATH.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def complete_top_three(
    predictions: list[list[str | None]], labels: list[str]
) -> list[list[str]]:
    """Fill baseline gaps with distinct allowed labels for valid rank-3 output."""
    completed = []
    for prediction in predictions:
        row = []
        for label in prediction:
            if label in labels and label not in row:
                row.append(label)
        row.extend(label for label in labels if label not in row)
        completed.append(row[:3])
    return completed


def works_name_stats() -> dict[str, float | int]:
    with (
        psycopg.connect(
            dsn(), options="-c default_transaction_read_only=on"
        ) as connection,
        connection.transaction(),
        connection.cursor() as cursor,
    ):
        cursor.execute("SET TRANSACTION READ ONLY")
        cursor.execute("SET LOCAL statement_timeout = 600000")
        cursor.execute(
            "SELECT count(*) AS rows, count(DISTINCT name) AS unique_names FROM public.works_names_mv"
        )
        rows, unique_names = cursor.fetchone()
    return {
        "rows": rows,
        "unique_work_names": unique_names,
        "duplication_factor": rows / unique_names,
    }


def main() -> int:
    ground_truth = read_ground_truth()
    examples = [row["source_work_name"] for row in ground_truth]
    labels = sorted({row["target_granular_name"] for row in ground_truth})
    if len(labels) != 466:
        raise RuntimeError(
            f"Expected 466 allowed labels, found {len(labels)}. Re-run the SAMPO audit before benchmarking."
        )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_csv(
        OUTPUT / "benchmark_inputs.csv",
        ["example_id", "raw_work_name"],
        [
            {"example_id": index, "raw_work_name": value}
            for index, value in enumerate(examples, start=1)
        ],
    )
    public_inputs = list(csv.DictReader((OUTPUT / "benchmark_inputs.csv").open(encoding="utf-8", newline="")))
    pilot_rows = random.Random(PILOT_SEED).sample(public_inputs, PILOT_SIZE)
    write_csv(OUTPUT / "pilot_inputs.csv", ["example_id", "raw_work_name"], pilot_rows)
    (OUTPUT / "pilot_manifest.json").write_text(
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
    write_csv(
        OUTPUT / "allowed_target_labels.csv",
        ["target_label"],
        [{"target_label": label} for label in labels],
    )
    results = {}
    for name, baseline in BASELINES.items():
        started = time.perf_counter()
        predictions = baseline(examples, labels, 3)
        elapsed = time.perf_counter() - started
        predictions = complete_top_three(predictions, labels)
        results[name] = {
            **evaluate_predictions(
                [
                    {"example_id": str(index), "target_granular_name": row["target_granular_name"]}
                    for index, row in enumerate(ground_truth, start=1)
                ],
                [
                    {"example_id": str(index), "top_1": row[0], "top_2": row[1], "top_3": row[2]}
                    for index, row in enumerate(predictions, start=1)
                ],
                labels,
            ),
            "runtime_seconds": elapsed,
        }
        write_csv(
            OUTPUT / f"predictions_{name}.csv",
            ["example_id", "top_1", "top_2", "top_3"],
            [
                {"example_id": index, "top_1": row[0], "top_2": row[1], "top_3": row[2]}
                for index, row in enumerate(predictions, start=1)
            ],
        )
    pilot_predictions = BASELINES["tfidf_char_ngrams"](
        [row["raw_work_name"] for row in pilot_rows], labels, 3
    )
    pilot_predictions = complete_top_three(pilot_predictions, labels)
    write_csv(
        OUTPUT / "predictions_tfidf_char_ngrams_pilot.csv",
        ["example_id", "top_1", "top_2", "top_3"],
        [
            {
                "example_id": row["example_id"],
                "top_1": prediction[0],
                "top_2": prediction[1],
                "top_3": prediction[2],
            }
            for row, prediction in zip(pilot_rows, pilot_predictions)
        ],
    )
    inference_source = (ROOT / "scripts" / "sampo_baselines.py").read_text(
        encoding="utf-8"
    )
    forbidden = [
        term
        for term in ("names_mapper", "granular_name", "private_ground_truth", "psycopg")
        if term in inference_source
    ]
    leakage = {
        "inference_module": "scripts/sampo_baselines.py",
        "public_inference_inputs": [
            "benchmark_inputs.csv",
            "allowed_target_labels.csv",
        ],
        "forbidden_terms_found_in_inference_source": forbidden,
        "passes": not forbidden,
        "private_gt_used_only_by": "scripts/build_sampo_benchmark.py evaluation path",
    }
    result = {
        "benchmark": {
            "examples": len(examples),
            "allowed_target_labels": len(labels),
            "targets_public": False,
        },
        "works_names_mv": works_name_stats(),
        "baselines": results,
        "leakage_check": leakage,
    }
    (OUTPUT / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (OUTPUT / "leakage_check.json").write_text(
        json.dumps(leakage, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = [
        "# SAMPO entity-resolution benchmark",
        "",
        f"- Examples: {len(examples):,}",
        f"- Public allowed target labels: {len(labels):,}",
        f"- Pilot: {PILOT_SIZE:,} deterministic random examples (seed `{PILOT_SEED}`)",
        f"- `works_names_mv`: {result['works_names_mv']['unique_work_names']:,} unique names across {result['works_names_mv']['rows']:,} rows (duplication factor {result['works_names_mv']['duplication_factor']:.2f})",
        "",
        "| Baseline | Top-1 | Top-3 | Macro-F1 (observed labels) | Runtime (s) |",
        "| --- | ---: | ---: | ---: | ---: |",
        *[
            f"| {name} | {scores['top_1_accuracy']:.4f} | {scores['top_3_accuracy']:.4f} | {scores['macro_f1_observed_labels']:.4f} | {scores['runtime_seconds']:.2f} |"
            for name, scores in results.items()
        ],
        "",
        f"Leakage check: {'passed' if leakage['passes'] else 'failed'}.",
    ]
    (OUTPUT / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote SAMPO benchmark to {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
