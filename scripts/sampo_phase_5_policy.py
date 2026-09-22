"""Trace-derived conformance checks for the generated Phase 5 policy."""
from __future__ import annotations

import json
import csv
import hashlib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "sampo_benchmark" / "candidate_artifacts"
RETRIEVERS = {"char_tfidf", "construction_token_tfidf", "word_tfidf"}


def _artifact(artifact_id: str) -> dict[str, Any]:
    return json.loads((ARTIFACTS / f"{artifact_id}.json").read_text(encoding="utf-8"))


def _artifact_id_from_prepare(call: dict[str, Any]) -> str | None:
    retrieval = call.get("retrieval") or {}
    try:
        with (ROOT / "artifacts" / "sampo_benchmark" / "pilot_inputs.csv").open(encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
        offset, limit = retrieval["offset"], retrieval["limit"]
        identity = {
            "pilot": rows[offset : offset + limit],
            "offset": offset,
            "limit": limit,
            "methods": sorted(retrieval["methods"]),
            "k": retrieval["k"],
            "fusion": retrieval["fusion"],
        }
        return hashlib.sha256(json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    except (KeyError, OSError, TypeError):
        return None


def _gate(artifact_id: str, assigned_ids: set[str]) -> tuple[set[str], set[str]]:
    data = _artifact(artifact_id)
    review: set[str] = set()
    for example in data["examples"]:
        if example["example_id"] not in assigned_ids:
            continue
        top1 = set()
        for entries in example["method_candidates"].values():
            top1.update(entry["method"] for entry in entries if entry["rank"] == 1)
        # The artifact contains one top-1 label per method. Reconstruct the
        # distinct labels rather than counting methods.
        labels = {
            label
            for label, entries in example["method_candidates"].items()
            if any(entry["rank"] == 1 for entry in entries)
        }
        if len(labels) > 1:
            review.add(example["example_id"])
    return review, assigned_ids - review


def evaluate_policy_conformance(
    trace: dict[str, Any], assigned_ids: list[str] | set[str]
) -> dict[str, Any]:
    """Verify complementary retrieval, disagreement gating, review, and writes."""
    assigned = set(assigned_ids)
    calls = trace.get("tool_calls", [])
    prepares = [call for call in calls if call.get("tool") == "prepare_candidate_batch"]
    valid_prepares = [
        call for call in prepares
        if set((call.get("retrieval") or {}).get("methods") or []) == RETRIEVERS
        and (call.get("retrieval") or {}).get("k") == 50
        and (call.get("retrieval") or {}).get("fusion") == "rrf"
    ]
    review_ids: set[str] = set()
    fallback_ids: set[str] = set()
    evidence_ids: set[str] = set()
    partition_ids: set[str] = set()
    reviewed_writes: set[str] = set()
    fallback_writes: set[str] = set()
    durable_seen: set[str] = set()
    gate_review: set[str] = set()
    gate_fallback: set[str] = set()
    reasons: list[str] = []

    if len(valid_prepares) != 1:
        reasons.append("expected exactly one complementary prepare_candidate_batch")
    prepare_positions = [i for i, call in enumerate(calls) if call.get("tool") == "prepare_candidate_batch"]
    evidence_positions = [i for i, call in enumerate(calls) if call.get("tool") == "get_candidate_evidence"]
    partition_calls = [call for call in calls if call.get("tool") == "partition_candidate_batch"]
    partition_positions = [i for i, call in enumerate(calls) if call.get("tool") == "partition_candidate_batch"]
    write_positions = [i for i, call in enumerate(calls) if call.get("tool") in {"save_review_decisions", "save_candidate_predictions"}]
    if prepare_positions and evidence_positions and min(evidence_positions) < max(prepare_positions):
        reasons.append("semantic evidence occurred before complementary retrieval completed")
    if write_positions and prepare_positions and min(write_positions) < max(prepare_positions):
        reasons.append("durable persistence occurred before complementary retrieval")
    if write_positions and evidence_positions and min(write_positions) < max(evidence_positions):
        reasons.append("durable persistence occurred before selective semantic review completed")
    if partition_positions and prepare_positions and min(partition_positions) < max(prepare_positions):
        reasons.append("disagreement gate occurred before complementary retrieval completed")
    if evidence_positions and partition_positions and min(evidence_positions) < max(partition_positions):
        reasons.append("semantic evidence occurred before disagreement gate")
    if write_positions:
        first_write = min(write_positions)
        last_write = max(write_positions)
        if any(call.get("tool") == "stage_candidate_predictions" for call in calls[first_write + 1:]):
            reasons.append("staging occurred after durable persistence")
        if any(call.get("tool") in {"prepare_candidate_batch", "get_candidate_evidence", "stage_candidate_predictions", "save_review_decisions", "save_candidate_predictions"} for call in calls[last_write + 1:]):
            reasons.append("MCP work continued after durable persistence")
    if valid_prepares:
        artifact_id = valid_prepares[0].get("artifact_id") or _artifact_id_from_prepare(valid_prepares[0])
        if not artifact_id:
            reasons.append("prepare_candidate_batch did not expose an artifact_id")
        else:
            try:
                gate_review, gate_fallback = _gate(artifact_id, assigned)
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                reasons.append(f"cannot reconstruct disagreement gate: {exc}")

    if len(partition_calls) != 1:
        reasons.append("expected exactly one deterministic partition_candidate_batch call")
    else:
        partition_ids = set(partition_calls[0].get("ids") or [])
        if partition_ids != assigned:
            reasons.append("partition_candidate_batch did not cover exactly the assigned IDs")

    for call in calls:
        tool = call.get("tool")
        ids = set(call.get("ids") or [])
        if tool == "get_candidate_evidence":
            evidence = call.get("evidence") or {}
            if evidence.get("selection") != "diverse_round_robin":
                reasons.append("review evidence did not use diverse_round_robin")
            if not 5 <= evidence.get("candidate_limit", 0) <= 10:
                reasons.append("review evidence candidate_limit was not 5--10")
            evidence_ids.update(ids)
        elif tool == "save_review_decisions":
            if durable_seen & ids:
                reasons.append("duplicate durable persistence attempt")
            reviewed_writes.update(ids)
            durable_seen.update(ids)
        elif tool == "save_candidate_predictions":
            if durable_seen & ids:
                reasons.append("duplicate durable persistence attempt")
            fallback_writes.update(ids)
            durable_seen.update(ids)

    review_ids = evidence_ids
    fallback_ids = assigned - review_ids
    if review_ids != gate_review:
        reasons.append("semantic-review IDs do not equal deterministic disagreement gate")
    if fallback_ids != gate_fallback:
        reasons.append("fallback IDs do not equal assigned IDs minus disagreement IDs")
    if reviewed_writes != gate_review:
        reasons.append("reviewed IDs were not persisted only through save_review_decisions")
    if fallback_writes != gate_fallback:
        reasons.append("fallback IDs were not persisted only through save_candidate_predictions")
    if reviewed_writes & fallback_writes:
        reasons.append("reviewed and fallback durable-write sets overlap")
    if reviewed_writes | fallback_writes != assigned:
        reasons.append("durable-write sets are not exhaustive")
    first_write = next((i for i, call in enumerate(calls) if call.get("tool") in {"save_review_decisions", "save_candidate_predictions"}), None)
    if first_write is not None and any(call.get("tool") == "get_candidate_evidence" for call in calls[first_write + 1:]):
        reasons.append("semantic evidence was requested after durable persistence")

    return {
        "pass": not reasons,
        "reasons": reasons,
        "expected_review_ids": sorted(gate_review),
        "expected_fallback_ids": sorted(gate_fallback),
        "evidence_ids": sorted(evidence_ids),
        "reviewed_write_ids": sorted(reviewed_writes),
        "fallback_write_ids": sorted(fallback_writes),
    }
