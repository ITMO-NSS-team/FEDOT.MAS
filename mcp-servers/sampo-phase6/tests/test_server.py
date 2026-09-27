from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mcp_sampo_phase6 import server as phase6_server


def _scope(monkeypatch, offset: int, size: int = 1) -> tuple[str, list[str]]:
    inputs = phase6_server.implementation._inputs("pilot_inputs.csv")
    ids = [row["example_id"] for row in inputs[offset : offset + size]]
    run_id = f"phase6_test_{offset}"
    monkeypatch.setenv("PHASE6_RUN_ID", run_id)
    monkeypatch.setenv("PHASE6_OFFSET", str(offset))
    monkeypatch.setenv("PHASE6_ASSIGNED_IDS", ",".join(ids))
    return run_id, ids


def _prediction(example_id: str, labels: list[str]) -> dict[str, str]:
    return {
        "example_id": example_id,
        "top_1": labels[0],
        "top_2": labels[1],
        "top_3": labels[2],
    }


def test_repeated_identical_prediction_is_idempotent_and_conflict_never_overwrites(
    tmp_path, monkeypatch
) -> None:
    run_id, ids = _scope(monkeypatch, 0)
    path = tmp_path / "predictions.jsonl"
    monkeypatch.setattr(phase6_server.implementation, "_run_path", lambda _: path)
    labels = phase6_server.implementation._labels()
    first = _prediction(ids[0], labels[:3])

    initial = phase6_server._persist_rows(run_id, [first])
    repeated = phase6_server._persist_rows(run_id, [first])

    assert initial["stored_ids"] == ids
    assert repeated["stored"] == 0
    assert repeated["already_identical_ids"] == ids
    conflict = _prediction(ids[0], [labels[1], labels[0], labels[2]])
    with pytest.raises(ValueError, match="conflicts with an existing stored top-3"):
        phase6_server._persist_rows(run_id, [conflict])
    assert [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] == [first]


def test_persistence_rejects_ids_outside_harness_batch(tmp_path, monkeypatch) -> None:
    run_id, assigned = _scope(monkeypatch, 0)
    outside = phase6_server.implementation._inputs("pilot_inputs.csv")[1]["example_id"]
    path = tmp_path / "predictions.jsonl"
    monkeypatch.setattr(phase6_server.implementation, "_run_path", lambda _: path)
    labels = phase6_server.implementation._labels()

    with pytest.raises(ValueError, match="current assigned batch"):
        phase6_server._persist_rows(run_id, [_prediction(outside, labels[:3])])
    assert assigned != [outside]
    assert not path.exists()


def test_batch_state_requires_matching_public_ids_and_run_id(monkeypatch) -> None:
    run_id, ids = _scope(monkeypatch, 0)
    assert phase6_server._batch() == (0, ids)
    phase6_server._run_id(run_id)
    monkeypatch.setenv("PHASE6_ASSIGNED_IDS", "not-an-assigned-id")
    with pytest.raises(ValueError, match="do not match the public pilot batch"):
        phase6_server._batch()
    with pytest.raises(ValueError, match="run_id must match"):
        phase6_server._run_id("another_run")
