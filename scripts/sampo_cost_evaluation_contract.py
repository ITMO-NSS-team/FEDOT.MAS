"""Artifact sealing and same-run evaluation helpers for the SAMPO cost demo."""
from __future__ import annotations

import csv
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SYSTEMS = ("tfidf", "fedotmas_cost_aware", "terra_single_agent")
PREDICTION_HEADER = ["example_id", "top_1", "top_2", "top_3"]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Missing or invalid artifact: {path}") from exc


def validate_and_seal(run_dir: Path, run_id: str, frozen_ids: list[str],
                      frozen_input_sha256: str, git_commit: str,
                      labels: set[str]) -> dict[str, Any]:
    manifest_path = run_dir / "experiment_manifest.json"
    manifest = _json(manifest_path)
    if manifest.get("run_id") != run_id:
        raise ValueError("experiment manifest run_id mismatch")
    systems: dict[str, Any] = {}
    expected_ids = set(frozen_ids)
    for system in SYSTEMS:
        directory = run_dir / system
        paths = {"predictions_csv": directory / "predictions.csv",
                 "telemetry_json": directory / "telemetry.json",
                 "metrics_json": directory / "metrics.json",
                 "runtime_manifest_json": directory / "runtime_manifest.json"}
        if system != "tfidf":
            paths["transcript_jsonl"] = directory / "transcript.jsonl"
        for path in paths.values():
            if not path.is_file():
                raise ValueError(f"Missing required artifact: {path}")
        with paths["predictions_csv"].open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != PREDICTION_HEADER:
                raise ValueError(f"Prediction schema must be exactly {','.join(PREDICTION_HEADER)}: {paths['predictions_csv']}")
            rows = list(reader)
        if any(None in row or set(row) != set(PREDICTION_HEADER) for row in rows):
            raise ValueError(f"Prediction row does not match the exact schema: {paths['predictions_csv']}")
        ids = [row["example_id"] for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate prediction IDs: {system}")
        if not set(ids) <= expected_ids:
            raise ValueError(f"Prediction IDs outside frozen run scope: {system}")
        for row in rows:
            ranked = [row.get(f"top_{rank}", "") for rank in (1, 2, 3)]
            if any(not label for label in ranked):
                raise ValueError(f"Prediction contains an empty label: {system}")
            if any(label not in labels for label in ranked):
                raise ValueError(f"Prediction contains a label outside the frozen label set: {system}")
            if len(set(ranked)) != 3:
                raise ValueError(f"Prediction contains duplicate ranked labels: {system}")
        metrics = _json(paths["metrics_json"])
        telemetry = _json(paths["telemetry_json"])
        runtime = _json(paths["runtime_manifest_json"])
        if metrics.get("run_id") != run_id:
            raise ValueError(f"{system} metrics run_id mismatch")
        if telemetry.get("run_id") != run_id or runtime.get("run_id") != run_id:
            raise ValueError(f"{system} artifact run_id mismatch")
        if metrics.get("stage") != "final" or telemetry.get("run_stage") != "final":
            raise ValueError(f"{system} artifact stage mismatch")
        if metrics.get("assigned_examples") != len(frozen_ids):
            raise ValueError(f"{system} metrics assigned scope mismatch")
        systems[system] = {key: sha256_file(path) for key, path in paths.items()}
    seal = {"run_id": run_id, "git_commit": git_commit,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "frozen_input_sha256": frozen_input_sha256,
            "experiment_manifest_sha256": sha256_file(manifest_path), "systems": systems}
    seal_path = run_dir / "sealed_predictions.json"
    if seal_path.exists():
        raise ValueError("Predictions already sealed for this run")
    seal_path.write_text(json.dumps(seal, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return seal


def resolve_final_scope(manifest: dict[str, Any], run_id: str, artifact_root: Path
                        ) -> tuple[Path, str, list[str]]:
    if manifest.get("run_id") != run_id:
        raise ValueError("experiment manifest run_id mismatch")
    if manifest.get("run_stage") != "final" or manifest.get("private_gt_evaluation_allowed") is not True:
        raise ValueError("Private GT evaluation is prohibited for this run stage")
    input_file = manifest.get("input_file")
    if not isinstance(input_file, str) or not input_file:
        raise ValueError("Run manifest is missing input_file")
    repo_root = artifact_root.parent.parent
    input_path = (repo_root / input_file).resolve()
    if not input_path.is_relative_to(repo_root.resolve()):
        raise ValueError("Run manifest input_file escapes repository root")
    expected_final = (artifact_root / "public_inputs.csv").resolve()
    if input_path != expected_final:
        raise ValueError("Final evaluation must use the frozen public_inputs.csv artifact")
    input_hash = manifest.get("input_sha256")
    if not isinstance(input_hash, str) or sha256_file(input_path) != input_hash:
        raise ValueError("Frozen public input SHA256 mismatch")
    with input_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    ids = [row.get("example_id", "") for row in rows]
    if not ids or any(not example_id for example_id in ids) or len(ids) != len(set(ids)):
        raise ValueError("Frozen run input contains empty or duplicate IDs")
    assigned_ids = manifest.get("assigned_ids")
    if not isinstance(assigned_ids, list) or assigned_ids != ids:
        raise ValueError("Frozen run scope does not match manifest assigned_ids")
    return input_path, input_hash, ids


def _read_private_gt_bytes(path: Path) -> bytes:
    return path.read_bytes()


def evaluate_run(run_dir: Path, run_id: str, manifest: dict[str, Any],
                 artifact_root: Path, private_gt_path: Path, labels: set[str],
                 evaluator_commit: str) -> dict[str, Any]:
    """Validate and seal public artifacts, verify the seal, then read private GT once."""
    _, frozen_input_hash, frozen_ids = resolve_final_scope(manifest, run_id, artifact_root)
    validate_and_seal(run_dir, run_id, frozen_ids, frozen_input_hash, evaluator_commit, labels)
    verify_seal(run_dir, run_id)
    gt_bytes = _read_private_gt_bytes(private_gt_path)
    return evaluate_once(run_dir, run_id, frozen_ids, gt_bytes, evaluator_commit, labels)


def verify_seal(run_dir: Path, run_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    seal = _json(run_dir / "sealed_predictions.json")
    if seal.get("run_id") != run_id:
        raise ValueError("sealed prediction run_id mismatch")
    for system, artifacts in seal["systems"].items():
        directory = run_dir / system
        for key, digest in artifacts.items():
            name = {"predictions_csv": "predictions.csv", "telemetry_json": "telemetry.json",
                    "metrics_json": "metrics.json", "runtime_manifest_json": "runtime_manifest.json",
                    "transcript_jsonl": "transcript.jsonl"}[key]
            path = directory / name
            if not path.is_file() or sha256_file(path) != digest:
                raise ValueError(f"Sealed artifact changed: {system}/{name}")
    if sha256_file(run_dir / "experiment_manifest.json") != seal["experiment_manifest_sha256"]:
        raise ValueError("Sealed experiment manifest changed")
    return seal, _json(run_dir / "experiment_manifest.json")


def evaluate_once(run_dir: Path, run_id: str, frozen_ids: list[str], gt_bytes: bytes,
                  evaluator_commit: str, labels: set[str]) -> dict[str, Any]:
    import csv as csv_module
    seal, _ = verify_seal(run_dir, run_id)
    gt_rows = list(csv_module.DictReader(io.StringIO(gt_bytes.decode("utf-8"))))
    gt_sha = sha256_bytes(gt_bytes)
    gt_by_id = {row["example_id"]: row["target_granular_name"] for row in gt_rows}
    if len(gt_by_id) != len(gt_rows) or set(frozen_ids) != set(gt_by_id):
        raise ValueError("Private GT IDs do not match the frozen run scope")
    systems: dict[str, Any] = {}
    predictions_by_system: dict[str, dict[str, dict[str, str]]] = {}
    for system in SYSTEMS:
        with (run_dir / system / "predictions.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        by_id = {row["example_id"]: row for row in rows}
        predictions_by_system[system] = by_id
        total = len(frozen_ids)
        correct1 = sum(by_id.get(eid, {}).get("top_1") == gt_by_id[eid] for eid in frozen_ids)
        correct3 = sum(gt_by_id[eid] in [by_id.get(eid, {}).get(f"top_{i}") for i in (1, 2, 3)] for eid in frozen_ids)
        completed = sum(bool(by_id.get(eid)) for eid in frozen_ids)
        metrics = _json(run_dir / system / "metrics.json")
        systems[system] = {"run_id": run_id, "system": system, "total_examples": total,
            "completed_examples": completed, "failed_or_missing_predictions": total - completed,
            "correct_top1_count": correct1, "correct_top3_count": correct3,
            "top1_accuracy": correct1 / total, "top3_accuracy": correct3 / total,
            "end_to_end_top1_accuracy": correct1 / total, "end_to_end_top3_accuracy": correct3 / total,
            "completed_only_top1_accuracy": correct1 / completed if completed else None,
            "completed_only_top3_accuracy": correct3 / completed if completed else None,
            "authoritative_total_cost_usd": metrics.get("authoritative_total_cost_usd"),
            "cost_complete": metrics.get("cost_complete")}
    return {"run_id": run_id, "private_gt_sha256": gt_sha, "_gt": gt_by_id, "_ids": frozen_ids,
            "sealed_predictions_sha256": sha256_file(run_dir / "sealed_predictions.json"),
            "evaluator_git_commit": evaluator_commit,
            "evaluation_timestamp": datetime.now(timezone.utc).isoformat(), "systems": systems,
            "_predictions": predictions_by_system}


def comparison(evaluation: dict[str, Any], run_dir: Path, paired_stats_fn) -> dict[str, Any]:
    run_id = evaluation["run_id"]
    seal, _ = verify_seal(run_dir, run_id)
    systems = evaluation["systems"]
    fed, terra = systems["fedotmas_cost_aware"], systems["terra_single_agent"]
    fcost, tcost = fed["authoritative_total_cost_usd"], terra["authoritative_total_cost_usd"]
    if not fed["cost_complete"] or fcost is None or not terra["cost_complete"] or tcost is None:
        raise ValueError("Missing live-system authoritative cost")
    predictions = evaluation["_predictions"]
    ids = evaluation["_ids"]
    stats = paired_stats_fn(ids, evaluation["_gt"],
                            predictions["fedotmas_cost_aware"], predictions["terra_single_agent"])
    gap = (terra["top1_accuracy"] - fed["top1_accuracy"]) * 100
    cost_correct_ratio = ((tcost / terra["correct_top1_count"]) /
                          (fcost / fed["correct_top1_count"])) if terra["correct_top1_count"] and fed["correct_top1_count"] and fcost else None
    cost_pass = (tcost / fcost >= 2.0) if fcost else False
    return {"run_id": run_id, "fedotmas_top1": fed["top1_accuracy"],
        "terra_top1": terra["top1_accuracy"], "fedotmas_top3": fed["top3_accuracy"],
        "terra_top3": terra["top3_accuracy"], "terra_minus_fedotmas_top1_gap_pp": gap,
        "relative_accuracy_retained": fed["top1_accuracy"] / terra["top1_accuracy"] if terra["top1_accuracy"] else None,
        "fedotmas_total_cost_usd": fcost, "terra_total_cost_usd": tcost,
        "terra_over_fedotmas_cost_ratio": tcost / fcost if fcost else None,
        "fedotmas_cost_per_correct_top1": fcost / fed["correct_top1_count"] if fed["correct_top1_count"] else None,
        "terra_cost_per_correct_top1": tcost / terra["correct_top1_count"] if terra["correct_top1_count"] else None,
        "cost_per_correct_ratio": cost_correct_ratio,
        "fedotmas_cost_per_1000_examples": fcost * 1000 / fed["total_examples"],
        "terra_cost_per_1000_examples": tcost * 1000 / terra["total_examples"],
        "both_correct": stats["both_correct"], "both_wrong": stats["both_wrong"],
        "fedotmas_only_correct": stats["a_only_correct"], "terra_only_correct": stats["b_only_correct"],
        "mcnemar_exact_two_sided_p": stats["mcnemar_exact_two_sided_p"],
        "bootstrap_ci_95_pp": stats["paired_bootstrap_95_ci_pp"],
        "sealed_prediction_hashes": seal["systems"],
        "accuracy_condition_pass": gap <= 3.0, "cost_condition_pass": cost_pass,
        "overall_primary_pass": gap <= 3.0 and cost_pass}
