"""Build and evaluate the private SAMPO entity-resolution benchmark."""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import Any

import psycopg
from sampo_baselines import BASELINES

ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = ROOT / "artifacts" / "sampo_audit"
OUTPUT = ROOT / "artifacts" / "sampo_benchmark"
GT_PATH = AUDIT_DIR / "private_ground_truth.csv"


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


def metrics(
    targets: list[str], predictions: list[list[str | None]], labels: list[str]
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
    targets = [row["target_granular_name"] for row in ground_truth]
    labels = sorted(set(targets))
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
        results[name] = {
            **metrics(targets, predictions, labels),
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
        f"- `works_names_mv`: {result['works_names_mv']['unique_work_names']:,} unique names across {result['works_names_mv']['rows']:,} rows (duplication factor {result['works_names_mv']['duplication_factor']:.2f})",
        "",
        "| Baseline | Top-1 | Top-3 | Macro-F1 | Runtime (s) |",
        "| --- | ---: | ---: | ---: | ---: |",
        *[
            f"| {name} | {scores['top_1_accuracy']:.4f} | {scores['top_3_accuracy']:.4f} | {scores['macro_f1']:.4f} | {scores['runtime_seconds']:.2f} |"
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
