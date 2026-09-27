from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from sampo_cost_demo import paired_stats
from sampo_cost_evaluation_contract import (
    comparison, evaluate_once, evaluate_run, resolve_final_scope, sha256_file,
    validate_and_seal, verify_seal,
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
        (directory / "telemetry.json").write_text(json.dumps({"run_id": RUN_ID, "run_stage": "final"}))
        if not (missing_metrics and system == "terra_single_agent"):
            cost = None if system == "terra_single_agent" else 0.2
            (directory / "metrics.json").write_text(json.dumps({"run_id": RUN_ID, "stage": "final", "assigned_examples": len(IDS),
                "authoritative_total_cost_usd": cost, "cost_complete": cost is not None}))
        (directory / "runtime_manifest.json").write_text(json.dumps({"run_id": RUN_ID}))
        if system != "tfidf":
            (directory / "transcript.jsonl").write_text("{}\n")
    return run_dir


def seal(run_dir: Path) -> None:
    validate_and_seal(run_dir, RUN_ID, IDS, "input-sha", "commit", {"a", "b", "c"})


def test_mismatched_run_ids_rejected(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path)
    with pytest.raises(ValueError, match="experiment manifest run_id mismatch"):
        validate_and_seal(run_dir, "another-run", IDS, "input-sha", "commit", {"a", "b", "c"})


def test_modified_prediction_after_sealing_rejected(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path)
    seal(run_dir)
    path = run_dir / "tfidf" / "predictions.csv"
    path.write_text(path.read_text() + "2,b,a,c\n")
    with pytest.raises(ValueError, match="Sealed artifact changed"):
        verify_seal(run_dir, RUN_ID)


def test_missing_metrics_rejected_before_sealing(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Missing required artifact: .*metrics.json"):
        validate_and_seal(make_run(tmp_path, missing_metrics=True), RUN_ID, IDS, "input-sha", "commit", {"a", "b", "c"})


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
        (run_dir / system / "metrics.json").write_text(json.dumps({"run_id": RUN_ID, "stage": "final", "assigned_examples": len(IDS),
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
        (run_dir / system / "metrics.json").write_text(json.dumps({"run_id": RUN_ID, "stage": "final", "assigned_examples": len(IDS),
            "authoritative_total_cost_usd": cost, "cost_complete": True}))
    seal(run_dir)
    evaluation = evaluate_once(run_dir, RUN_ID, IDS,
        b"example_id,target_granular_name\n1,a\n2,b\n", "commit", {"a", "b", "c"})
    result = comparison(evaluation, run_dir, paired_stats)
    for system in SYSTEMS:
        assert result["sealed_prediction_hashes"][system]["predictions_csv"] == sha256_file(
            run_dir / system / "predictions.csv")


def make_scoped_run(tmp_path: Path, stage: str = "final") -> tuple[Path, Path, dict]:
    repo = tmp_path / "repo"
    artifact_root = repo / "artifacts" / "sampo_cost_demo"
    (artifact_root / "runs").mkdir(parents=True)
    final_path = artifact_root / "public_inputs.csv"
    final_path.write_text("example_id,raw_work_name\n1,x\n2,y\n")
    smoke_path = artifact_root / "operational_inputs.csv"
    smoke_path.write_text("example_id,raw_work_name\n8,u\n9,v\n")
    run_dir = make_run(artifact_root / "runs")
    import hashlib
    public_path = final_path if stage == "final" else smoke_path
    manifest = {"run_id": RUN_ID, "run_stage": stage,
        "input_file": public_path.relative_to(repo).as_posix(),
        "input_sha256": hashlib.sha256(public_path.read_bytes()).hexdigest(),
        "assigned_ids": IDS if stage == "final" else ["8", "9"],
        "private_gt_evaluation_allowed": stage == "final"}
    (run_dir / "experiment_manifest.json").write_text(json.dumps(manifest))
    return run_dir, artifact_root, manifest


def fail_if_gt_opened(monkeypatch: pytest.MonkeyPatch) -> None:
    import sampo_cost_evaluation_contract as contract
    monkeypatch.setattr(contract, "_read_private_gt_bytes",
                        lambda _path: pytest.fail("private GT was opened"))


def test_smoke_manifest_cannot_be_evaluated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir, artifact_root, manifest = make_scoped_run(tmp_path, "smoke")
    fail_if_gt_opened(monkeypatch)
    with pytest.raises(ValueError, match="prohibited"):
        evaluate_run(run_dir, RUN_ID, manifest, artifact_root, artifact_root / "private_ground_truth.csv",
                     {"a", "b", "c"}, "commit")


def test_final_manifest_resolves_frozen_public_inputs(tmp_path: Path) -> None:
    _, artifact_root, manifest = make_scoped_run(tmp_path)
    input_path, digest, ids = resolve_final_scope(manifest, RUN_ID, artifact_root)
    assert input_path == artifact_root / "public_inputs.csv"
    assert input_path != artifact_root / "operational_inputs.csv"
    assert digest == manifest["input_sha256"]
    assert ids == IDS


def test_input_sha_mismatch_fails_before_gt_access(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir, artifact_root, manifest = make_scoped_run(tmp_path)
    manifest["input_sha256"] = "bad-hash"
    fail_if_gt_opened(monkeypatch)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        evaluate_run(run_dir, RUN_ID, manifest, artifact_root, artifact_root / "private_ground_truth.csv",
                     {"a", "b", "c"}, "commit")


def test_seal_verification_failure_prevents_gt_access(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir, artifact_root, manifest = make_scoped_run(tmp_path)
    import sampo_cost_evaluation_contract as contract
    fail_if_gt_opened(monkeypatch)
    monkeypatch.setattr(contract, "verify_seal", lambda *_args: (_ for _ in ()).throw(ValueError("seal verification failed")))
    with pytest.raises(ValueError, match="seal verification failed"):
        evaluate_run(run_dir, RUN_ID, manifest, artifact_root, artifact_root / "private_ground_truth.csv",
                     {"a", "b", "c"}, "commit")


@pytest.mark.parametrize("bad_row", [
    {"example_id": "1", "top_1": "bad", "top_2": "b", "top_3": "c"},
    {"example_id": "outside", "top_1": "a", "top_2": "b", "top_3": "c"},
])
def test_invalid_or_out_of_vocabulary_prediction_fails_before_gt_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad_row: dict[str, str],
) -> None:
    run_dir, artifact_root, manifest = make_scoped_run(tmp_path)
    path = run_dir / "tfidf" / "predictions.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["example_id", "top_1", "top_2", "top_3"])
        writer.writeheader()
        writer.writerow(bad_row)
    fail_if_gt_opened(monkeypatch)
    with pytest.raises(ValueError):
        evaluate_run(run_dir, RUN_ID, manifest, artifact_root, artifact_root / "private_ground_truth.csv",
                     {"a", "b", "c"}, "commit")


def test_duplicate_top3_fails_before_gt_access(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir, artifact_root, manifest = make_scoped_run(tmp_path)
    path = run_dir / "tfidf" / "predictions.csv"
    path.write_text("example_id,top_1,top_2,top_3\n1,a,b,a\n")
    fail_if_gt_opened(monkeypatch)
    with pytest.raises(ValueError, match="duplicate ranked labels"):
        evaluate_run(run_dir, RUN_ID, manifest, artifact_root, artifact_root / "private_ground_truth.csv",
                     {"a", "b", "c"}, "commit")


def test_final_gt_ids_must_exactly_match_run_scope(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path)
    seal(run_dir)
    with pytest.raises(ValueError, match="do not match the frozen run scope"):
        evaluate_once(run_dir, RUN_ID, IDS, b"example_id,target_granular_name\n1,a\n", "commit", {"a", "b", "c"})


def test_same_run_accuracy_cost_and_criteria_pass(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path)
    for system in SYSTEMS:
        path = run_dir / system / "predictions.csv"
        path.write_text("example_id,top_1,top_2,top_3\n1,a,b,c\n2,b,a,c\n")
    for system, cost in (("fedotmas_cost_aware", 0.1), ("terra_single_agent", 0.2)):
        (run_dir / system / "metrics.json").write_text(json.dumps({"run_id": RUN_ID, "stage": "final", "assigned_examples": len(IDS),
            "authoritative_total_cost_usd": cost, "cost_complete": True}))
    seal(run_dir)
    evaluation = evaluate_once(run_dir, RUN_ID, IDS,
        b"example_id,target_granular_name\n1,a\n2,b\n", "commit", {"a", "b", "c"})
    result = comparison(evaluation, run_dir, paired_stats)
    assert result["terra_minus_fedotmas_top1_gap_pp"] == 0
    assert result["terra_over_fedotmas_cost_ratio"] == 2
    assert result["accuracy_condition_pass"] is True
    assert result["cost_condition_pass"] is True
    assert result["overall_primary_pass"] is True
