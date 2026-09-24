"""Batch-scoped neutral wrappers around the public SAMPO benchmark tools."""
from __future__ import annotations

import os
import sys
import fcntl
import json
import re
from pathlib import Path
from typing import Any

from fastmcp import FastMCP
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[4]
LEGACY_SOURCE = ROOT / "mcp-servers" / "sampo-benchmark" / "src"
sys.path.insert(0, str(LEGACY_SOURCE))
from mcp_sampo_benchmark import server as implementation  # noqa: E402

mcp = FastMCP("sampo-phase6")
MAX_INSPECT_IDS = 10
MAX_INSPECT_CANDIDATES = 30


class RankedTop3(BaseModel):
    """Three artifact-local indices for one example."""

    example_id: str = Field(description="Example ID from the current batch")
    candidate_indices: list[int] = Field(
        min_length=3,
        max_length=3,
        description="Three distinct artifact-local candidate indices in rank order",
    )


def _batch() -> tuple[int, list[str]]:
    raw_ids = os.environ.get("PHASE6_ASSIGNED_IDS", "")
    try:
        offset = int(os.environ["PHASE6_OFFSET"])
        ids = [item for item in raw_ids.split(",") if item]
    except (KeyError, ValueError) as exc:
        raise ValueError("The current batch is not configured") from exc
    if offset < 0 or not ids or len(ids) > 20 or len(ids) != len(set(ids)):
        raise ValueError("The configured batch must contain 1-20 unique IDs")
    run_id = os.environ.get("PHASE6_RUN_ID", "")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", run_id):
        raise ValueError("The current run is not configured")
    expected_ids = [
        row["example_id"]
        for row in implementation._inputs("pilot_inputs.csv")[offset : offset + len(ids)]
    ]
    if ids != expected_ids:
        raise ValueError("The configured IDs do not match the public pilot batch")
    return offset, ids


def _artifact_for_batch(artifact_id: str) -> dict[str, Any]:
    offset, ids = _batch()
    data = implementation._artifact(artifact_id)
    parameters = data.get("parameters", {})
    artifact_ids = [row.get("example_id") for row in data.get("examples", [])]
    if (
        parameters.get("offset") != offset
        or parameters.get("limit") != len(ids)
        or artifact_ids != ids
        or data.get("artifact_id") != artifact_id
    ):
        raise ValueError("Candidate artifact does not match the current batch")
    return data


def _run_id(supplied: str) -> None:
    expected = os.environ.get("PHASE6_RUN_ID")
    if not expected or supplied != expected:
        raise ValueError("run_id must match the current assigned batch")


def _prediction_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("example_id", "")),
        str(row.get("top_1", "")),
        str(row.get("top_2", "")),
        str(row.get("top_3", "")),
    )


def _persist_rows(run_id: str, rows: list[dict[str, str]]) -> dict[str, Any]:
    """Append new predictions and treat exact repeated predictions as no-ops."""
    _, assigned = _batch()
    if not rows:
        raise ValueError("At least one prediction is required")
    incoming_ids = [row.get("example_id", "") for row in rows]
    if len(incoming_ids) != len(set(incoming_ids)):
        raise ValueError("Prediction IDs must be unique within a write")
    if not set(incoming_ids) <= set(assigned):
        raise ValueError("Prediction IDs must belong to the current assigned batch")
    implementation._valid(rows)

    path = implementation._run_path(run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        try:
            file.seek(0)
            stored_rows = [json.loads(line) for line in file if line.strip()]
            stored_by_id = {row.get("example_id"): row for row in stored_rows}
            if len(stored_by_id) != len(stored_rows):
                raise ValueError("Stored run contains duplicate prediction IDs")
            if not set(stored_by_id) <= set(assigned):
                raise ValueError("Stored run contains IDs outside the current assigned batch")

            new_rows: list[dict[str, str]] = []
            identical_ids: list[str] = []
            conflicts: list[str] = []
            for row in rows:
                example_id = row["example_id"]
                current = stored_by_id.get(example_id)
                if current is None:
                    new_rows.append(row)
                elif _prediction_key(current) == _prediction_key(row):
                    identical_ids.append(example_id)
                else:
                    conflicts.append(example_id)
            if conflicts:
                raise ValueError(
                    "Prediction conflicts with an existing stored top-3 for IDs: "
                    + ", ".join(conflicts)
                )

            file.seek(0, os.SEEK_END)
            for row in new_rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
            if new_rows:
                file.flush()
                os.fsync(file.fileno())
            return {
                "stored": len(new_rows),
                "stored_ids": [row["example_id"] for row in new_rows],
                "already_identical": len(identical_ids),
                "already_identical_ids": identical_ids,
                "total": len(stored_by_id) + len(new_rows),
            }
        finally:
            fcntl.flock(file, fcntl.LOCK_UN)


@mcp.tool
def list_methods() -> dict[str, Any]:
    """Return supported method names, aliases, fusion values, limits, and allowed labels."""
    _batch()
    return {**implementation.list_methods(), "allowed_target_labels": implementation._labels()}


@mcp.tool
def prepare_candidates(
    offset: int, limit: int, methods: list[str], k: int, fusion: str
) -> dict[str, Any]:
    """Create or load candidate results and public work names for a batch. Parameters select methods, result count, and fusion."""
    expected_offset, ids = _batch()
    if offset != expected_offset or limit != len(ids):
        raise ValueError("offset and limit must match the current assigned batch")
    result = implementation.prepare_candidate_batch(offset, limit, methods, k, fusion)
    names = {
        row["example_id"]: row["raw_work_name"]
        for row in implementation._inputs("pilot_inputs.csv")[offset : offset + limit]
    }
    for example in result["examples"]:
        example["raw_work_name"] = names[example["example_id"]]
    return result


@mcp.tool
def inspect_candidates(
    artifact_id: str,
    example_ids: list[str],
    candidate_indices: list[int] | None = None,
    candidate_limit: int | None = None,
    selection: str = "fused",
) -> dict[str, Any]:
    """Return candidate rows and method-level data for supplied IDs and candidate indices. Accepts 1-10 IDs and at most 30 candidates per ID."""
    if not 1 <= len(example_ids) <= MAX_INSPECT_IDS or len(set(example_ids)) != len(example_ids):
        raise ValueError(f"Provide 1-{MAX_INSPECT_IDS} unique example IDs")
    if candidate_indices is not None:
        if not candidate_indices or len(candidate_indices) > MAX_INSPECT_CANDIDATES:
            raise ValueError(f"candidate_indices must contain 1-{MAX_INSPECT_CANDIDATES} values")
        if any(not isinstance(index, int) or not 0 <= index < MAX_INSPECT_CANDIDATES for index in candidate_indices):
            raise ValueError(f"candidate_indices must be integers from 0-{MAX_INSPECT_CANDIDATES - 1}")
        if len(candidate_indices) != len(set(candidate_indices)):
            raise ValueError("candidate_indices must be unique")
    if candidate_limit is not None and not 1 <= candidate_limit <= MAX_INSPECT_CANDIDATES:
        raise ValueError(f"candidate_limit must be 1-{MAX_INSPECT_CANDIDATES}")
    if candidate_indices is not None and candidate_limit is not None:
        raise ValueError("Use candidate_indices or candidate_limit, not both")
    data = _artifact_for_batch(artifact_id)
    batch_ids = {row["example_id"] for row in data["examples"]}
    if not set(example_ids) <= batch_ids:
        raise ValueError("IDs must belong to the current batch artifact")

    effective_limit = (
        max(candidate_indices) + 1
        if candidate_indices is not None
        else candidate_limit or MAX_INSPECT_CANDIDATES
    )
    result = implementation.get_candidate_evidence(
        artifact_id,
        example_ids,
        candidate_limit=effective_limit,
        selection=selection,
    )
    if candidate_indices is not None:
        requested = set(candidate_indices)
        for example in result["examples"]:
            candidates = [
                row for row in example["fused_candidates"]
                if row["candidate_index"] in requested
            ]
            if len(candidates) != len(requested):
                raise ValueError("A candidate index does not belong to an example")
            example["fused_candidates"] = candidates
            labels = {row["label"] for row in candidates}
            example["method_candidates"] = {
                label: entries
                for label, entries in example["method_candidates"].items()
                if label in labels
            }
    return result


@mcp.tool
def save_default_top3(run_id: str, artifact_id: str, example_ids: list[str]) -> dict[str, Any]:
    """Persist the artifact's default top-three for supplied IDs. Predictions are write-once. An exact repeated prediction is an idempotent no-op; a different prediction for an already stored ID is rejected."""
    _run_id(run_id)
    data = _artifact_for_batch(artifact_id)
    if not example_ids or len(example_ids) != len(set(example_ids)):
        raise ValueError("Provide unique example IDs")
    if not set(example_ids) <= {row["example_id"] for row in data["examples"]}:
        raise ValueError("IDs must belong to the current batch artifact")
    known = {row["example_id"]: row for row in data["examples"]}
    rows: list[dict[str, str]] = []
    for example_id in example_ids:
        candidates = known[example_id].get("fused_candidates", [])
        if len(candidates) < 3:
            raise ValueError("Artifact must contain at least three default candidates per ID")
        rows.append({
            "example_id": example_id,
            **{f"top_{rank}": candidates[rank - 1]["label"] for rank in range(1, 4)},
        })
    return _persist_rows(run_id, rows)


@mcp.tool
def save_ranked_top3(
    run_id: str, artifact_id: str, rankings: list[RankedTop3]
) -> dict[str, Any]:
    """Persist caller-selected top-three indices. Predictions are write-once. An exact repeated prediction is an idempotent no-op; a different prediction for an already stored ID is rejected."""
    _run_id(run_id)
    data = _artifact_for_batch(artifact_id)
    decisions = [ranking.model_dump() for ranking in rankings]
    decision_ids = [item["example_id"] for item in decisions]
    if not decisions or len(decision_ids) != len(set(decision_ids)):
        raise ValueError("Provide one ranking per unique example ID")
    known = {row["example_id"]: row for row in data["examples"]}
    if not set(decision_ids) <= set(known):
        raise ValueError("IDs must belong to the current batch artifact")
    rows: list[dict[str, str]] = []
    for decision in decisions:
        example_id = decision["example_id"]
        indices = decision["candidate_indices"]
        if len(indices) != 3 or len(set(indices)) != 3 or any(type(i) is not int for i in indices):
            raise ValueError("Each ranking needs three distinct candidate indices")
        by_index = {
            candidate["candidate_index"]: candidate["label"]
            for candidate in known[example_id].get("fused_candidates", [])
        }
        if any(index not in by_index for index in indices):
            raise ValueError("Candidate index does not belong to the supplied example")
        labels = [by_index[index] for index in indices]
        if len(set(labels)) != 3:
            raise ValueError("Candidate indices must identify three distinct labels")
        rows.append({
            "example_id": example_id,
            **{f"top_{rank}": label for rank, label in enumerate(labels, start=1)},
        })
    return _persist_rows(run_id, rows)


@mcp.tool
def get_prediction_status(run_id: str, example_ids: list[str]) -> dict[str, Any]:
    """Return stored and missing status for 1-20 supplied IDs from the current batch."""
    _run_id(run_id)
    _, assigned = _batch()
    if not 1 <= len(example_ids) <= 20 or len(set(example_ids)) != len(example_ids):
        raise ValueError("Provide 1-20 unique example IDs")
    if not set(example_ids) <= set(assigned):
        raise ValueError("IDs must belong to the current assigned batch")
    return implementation.get_prediction_status(run_id, example_ids)


def main() -> None:
    mcp.run(show_banner=False)
