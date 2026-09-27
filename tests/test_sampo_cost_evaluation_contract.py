from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from sampo_cost_demo import paired_stats
from sampo_cost_evaluation_contract import (
    comparison, evaluate_once, sha256_file, validate_and_seal, verify_seal,
)


RUN_ID = "contract-test-run"
IDS = ["1", "2"]
SYSTEMS = ("tfidf", "fedotmas_cost_aware", "terra_single_agent")


def make_run(tmp_path: Path, *, missing_metrics: bool = False) -> Path:
    run_dir = tmp_path / RUN_ID
    run_dir.mkdir()
    (run_dir / "experiment_manifest.json").write_text(json.dumps({"run_id": RUN_ID}))
    for system in SYSTEMS:
        directory = run_dir / system
        directory.mkdir()
        with (directory / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["example_id", "top_1", "top_2", "top_3"])
            writer.writeheader()
            if system != "terra_single_agent":
                writer.writerow({"example_id": "1", "top_1": "a", "top_2": "b", "top_3": "c"})
        (directory / "telemetry.json").write_text(json.dumps({"run_id": RUN_ID}))
        if not (missing_metrics and system == "terra_single_agent"):
            cost = None if system == "terra_single_agent" else 0.2
            (directory / "metrics.json").write_text(json.dumps({"run_id": RUN_ID,
                "authoritative_total_cost_usd": cost, "cost_complete": cost is not None}))
        (directory / "runtime_manifest.json").write_text(json.dumps({"run_id": RUN_ID}))
        if system != "tfidf":
            (directory / "transcript.jsonl").write_text("{}\n")
    return run_dir


def seal(run_dir: Path) -> None:
    validate_and_seal(run_dir, RUN_ID, IDS, "input-sha", "commit")


def test_mismatched_run_ids_rejected(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path)
    with pytest.raises(ValueError, match="experiment manifest run_id mismatch"):
        validate_and_seal(run_dir, "another-run", IDS, "input-sha", "commit")


def test_modified_prediction_after_sealing_rejected(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path)
    seal(run_dir)
    path = run_dir / "tfidf" / "predictions.csv"
    path.write_text(path.read_text() + "2,b,a,c\n")
    with pytest.raises(ValueError, match="Sealed artifact changed"):
        verify_seal(run_dir, RUN_ID)


def test_missing_metrics_rejected_before_sealing(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Missing required artifact: .*metrics.json"):
        validate_and_seal(make_run(tmp_path, missing_metrics=True), RUN_ID, IDS, "input-sha", "commit")


def test_missing_live_system_cost_rejected(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path)
    seal(run_dir)
    evaluation = evaluate_once(run_dir, RUN_ID, IDS,
        b"example_id,target_granular_name\n1,a\n2,b\n", "commit", {"a", "b", "c"})
    with pytest.raises(ValueError, match="Missing live-system authoritative cost"):
        comparison(evaluation, run_dir, paired_stats)


def test_failed_predictions_stay_in_end_to_end_denominator(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path)
    # Make live costs complete while keeping Terra's missing predictions.
    for system, cost in (("fedotmas_cost_aware", 0.2), ("terra_single_agent", 0.3)):
        (run_dir / system / "metrics.json").write_text(json.dumps({"run_id": RUN_ID,
            "authoritative_total_cost_usd": cost, "cost_complete": True}))
    seal(run_dir)
    evaluation = evaluate_once(run_dir, RUN_ID, IDS,
        b"example_id,target_granular_name\n1,a\n2,b\n", "commit", {"a", "b", "c"})
    assert evaluation["systems"]["terra_single_agent"]["total_examples"] == 2
    assert evaluation["systems"]["terra_single_agent"]["failed_or_missing_predictions"] == 2
    assert evaluation["systems"]["terra_single_agent"]["end_to_end_top1_accuracy"] == 0


def test_comparison_carries_exact_sealed_prediction_hashes(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path)
    for system, cost in (("fedotmas_cost_aware", 0.2), ("terra_single_agent", 0.3)):
        (run_dir / system / "metrics.json").write_text(json.dumps({"run_id": RUN_ID,
            "authoritative_total_cost_usd": cost, "cost_complete": True}))
    seal(run_dir)
    evaluation = evaluate_once(run_dir, RUN_ID, IDS,
        b"example_id,target_granular_name\n1,a\n2,b\n", "commit", {"a", "b", "c"})
    result = comparison(evaluation, run_dir, paired_stats)
    for system in SYSTEMS:
        assert result["sealed_prediction_hashes"][system]["predictions_csv"] == sha256_file(
            run_dir / system / "predictions.csv")
