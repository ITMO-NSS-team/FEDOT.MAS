#!/usr/bin/env python3
"""Seal public inference artifacts, then perform the single private-GT evaluation."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sampo_cost_demo import OUT, read_csv, paired_stats
from sampo_cost_evaluation_contract import (
    comparison, evaluate_run,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    run_dir = OUT / "runs" / args.run_id
    if not run_dir.is_dir():
        raise SystemExit(f"Unknown run ID: {args.run_id}")
    for output in ("sealed_predictions.json", "evaluation.json", "comparison.json"):
        if (run_dir / output).exists():
            raise SystemExit(f"{output} already exists; evaluation is one-shot")

    manifest_path = run_dir / "experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("run_id") != args.run_id:
        raise SystemExit("Experiment manifest run_id mismatch")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=OUT.parents[1], text=True).strip()
    labels = {row["target_label"] for row in read_csv(OUT / "allowed_target_labels.csv")}
    evaluation = evaluate_run(run_dir, args.run_id, manifest, OUT,
                              OUT / "private_ground_truth.csv", labels, commit)
    gt_mapping = evaluation.pop("_gt")
    predictions = evaluation.pop("_predictions")
    ids = evaluation.pop("_ids")
    (run_dir / "evaluation.json").write_text(json.dumps(evaluation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    comparison_input = {**evaluation, "_gt": gt_mapping, "_predictions": predictions, "_ids": ids}
    summary = comparison(comparison_input, run_dir, paired_stats)
    (run_dir / "comparison.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Backward-compatible root reports are views derived from the authoritative run files.
    report = {"run_id": args.run_id, "evaluation": evaluation, "comparison": summary}
    (run_dir / "final_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
