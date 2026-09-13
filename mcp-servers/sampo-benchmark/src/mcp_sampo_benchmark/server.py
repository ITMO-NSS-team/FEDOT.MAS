from __future__ import annotations

import csv
import fcntl
import json
import re
import sys
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

ROOT = Path(__file__).resolve().parents[4]
PUBLIC = ROOT / "artifacts" / "sampo_benchmark"
sys.path.insert(0, str(ROOT / "scripts"))
from sampo_baselines import (
    bm25_token_ranked,
    tfidf_char_ngrams_ranked,
    tfidf_char_word_hybrid_ranked,
    tfidf_construction_token_ranked,
    tfidf_word_ranked,
)

mcp = FastMCP("sampo-benchmark")
RETRIEVERS = {
    "bm25_token": bm25_token_ranked,
    "char_tfidf": tfidf_char_ngrams_ranked,
    "char_word_fusion": tfidf_char_word_hybrid_ranked,
    "construction_token_tfidf": tfidf_construction_token_ranked,
    "word_tfidf": tfidf_word_ranked,
}


def _inputs(filename: str = "benchmark_inputs.csv") -> list[dict[str, str]]:
    if filename not in {"benchmark_inputs.csv", "pilot_inputs.csv"}:
        raise ValueError("Only benchmark_inputs.csv and pilot_inputs.csv are available")
    with (PUBLIC / filename).open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _labels() -> list[str]:
    with (PUBLIC / "allowed_target_labels.csv").open(
        encoding="utf-8", newline=""
    ) as file:
        return [row["target_label"] for row in csv.DictReader(file)]


@mcp.tool
def list_methods() -> dict[str, Any]:
    """List independent public candidate-retrieval methods and their limits.

    Methods are alternatives with method-local scores. Retrieve one or more
    methods over the same bounded ID batch when complementary evidence is useful.
    """
    return {
        "methods": sorted(RETRIEVERS),
        "max_batch_size": 100,
        "max_k": 50,
        "result_fields": ["example_id", "label", "rank", "score"],
    }


@mcp.tool
def retrieve_candidates(
    example_ids: list[str], method: str, k: int = 5
) -> dict[str, Any]:
    """Retrieve one method's ranked candidates for a bounded batch of pilot IDs.

    Every result includes rank and a score comparable only within the requested
    method. Call this uniform API separately for any methods to be considered.
    """
    if method not in RETRIEVERS:
        raise ValueError("Unknown retrieval method")
    if len(example_ids) > 100 or not 1 <= k <= 50:
        raise ValueError("At most 100 IDs and k <= 50")
    source = {row["example_id"]: row for row in _inputs("pilot_inputs.csv")}
    if len(set(example_ids)) != len(example_ids) or any(
        item not in source for item in example_ids
    ):
        raise ValueError("IDs must be unique fixed-pilot IDs")
    rankings = RETRIEVERS[method](
        [source[item]["raw_work_name"] for item in example_ids], _labels(), k
    )
    return {
        "method": method,
        "candidates": [
            {
                "example_id": item,
                "candidates": [
                    {"label": label, "rank": rank, "score": score}
                    for rank, (label, score) in enumerate(ranking, 1)
                ],
            }
            for item, ranking in zip(example_ids, rankings)
        ],
    }


def _run_path(run_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", run_id):
        raise ValueError("Invalid run_id")
    return PUBLIC / "mas_runs" / f"{run_id}.jsonl"


def _validate_rows(
    rows: list[dict[str, Any]], expected: set[str], labels: set[str]
) -> None:
    if len({row.get("example_id") for row in rows}) != len(rows):
        raise ValueError("Batch contains duplicate IDs")
    for row in rows:
        values = [row.get(name) for name in ("top_1", "top_2", "top_3")]
        if (
            row.get("example_id") not in expected
            or any(value not in labels for value in values)
            or len(set(values)) != 3
        ):
            raise ValueError("Invalid or non-pilot prediction")


@mcp.tool
def save_prediction_batch(rows: list[dict[str, Any]], run_id: str) -> dict[str, Any]:
    """Persist rows with exact schema [{example_id, top_1, top_2, top_3}] in shared pilot storage."""
    expected = {row["example_id"] for row in _inputs("pilot_inputs.csv")}
    labels = set(_labels())
    path = _run_path(run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        file.seek(0)
        prior = [json.loads(line) for line in file if line.strip()]
        seen = {row["example_id"] for row in prior}
        _validate_rows(rows, expected, labels)
        if seen & {row["example_id"] for row in rows}:
            raise ValueError(
                "Run already contains a prediction ID; use replace_prediction_batch for revisions"
            )
        seen.update(row["example_id"] for row in rows)
        file.seek(0, 2)
        file.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        fcntl.flock(file, fcntl.LOCK_UN)
    return {"saved": len(rows), "total": len(seen)}


@mcp.tool
def import_prediction_file(filename: str, run_id: str) -> dict[str, Any]:
    """Import valid public CSV rows, filling only IDs absent from the shared run."""
    source = PUBLIC / "mas_runs" / Path(filename).name
    if not source.exists() or source.suffix != ".csv":
        raise ValueError("Only an existing CSV in public mas_runs may be imported")
    with source.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    expected = {row["example_id"] for row in _inputs("pilot_inputs.csv")}
    labels = set(_labels())
    path = _run_path(run_id)
    _validate_rows(rows, expected, labels)
    with path.open("a+", encoding="utf-8") as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        file.seek(0)
        stored = {
            row["example_id"]: row
            for line in file
            if line.strip()
            for row in [json.loads(line)]
        }
        additions = [row for row in rows if row["example_id"] not in stored]
        stored.update({row["example_id"]: row for row in additions})
        file.seek(0)
        file.truncate()
        file.writelines(
            json.dumps(row, ensure_ascii=False) + "\n" for row in stored.values()
        )
        fcntl.flock(file, fcntl.LOCK_UN)
    return {
        "imported": len(additions),
        "preserved": len(rows) - len(additions),
        "total": len(stored),
    }


@mcp.tool
def replace_prediction_batch(rows: list[dict[str, Any]], run_id: str) -> dict[str, Any]:
    """Atomically replace existing predictions for IDs within one shared pilot run."""
    expected = {row["example_id"] for row in _inputs("pilot_inputs.csv")}
    labels = set(_labels())
    path = _run_path(run_id)
    with path.open("a+", encoding="utf-8") as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        file.seek(0)
        stored = {
            row["example_id"]: row
            for line in file
            if line.strip()
            for row in [json.loads(line)]
        }
        _validate_rows(rows, expected, labels)
        stored.update({row["example_id"]: row for row in rows})
        file.seek(0)
        file.truncate()
        file.writelines(
            json.dumps(row, ensure_ascii=False) + "\n" for row in stored.values()
        )
        fcntl.flock(file, fcntl.LOCK_UN)
    return {"replaced": len(rows), "total": len(stored)}


@mcp.tool
def finalize_predictions(run_id: str) -> dict[str, Any]:
    """Finalize only when every fixed-pilot ID appears exactly once."""
    path = _run_path(run_id)
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    expected = {row["example_id"] for row in _inputs("pilot_inputs.csv")}
    if len(rows) != len(expected) or {row["example_id"] for row in rows} != expected:
        raise ValueError("Run does not contain every pilot ID exactly once")
    output = path.with_suffix(".csv")
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=["example_id", "top_1", "top_2", "top_3"]
        )
        writer.writeheader()
        writer.writerows(rows)
    return {"prediction_path": str(output.relative_to(ROOT)), "examples": len(rows)}


@mcp.tool
def get_allowed_labels() -> dict[str, Any]:
    """Return the complete public list of allowed granular work labels."""
    labels = _labels()
    return {"labels": labels, "count": len(labels)}


@mcp.tool
def get_input_batch(offset: int = 0, limit: int = 100) -> dict[str, Any]:
    """Return a bounded public batch of historical work names; limit is capped at 200."""
    rows = _inputs()
    offset, limit = max(offset, 0), min(max(limit, 1), 200)
    return {
        "examples": rows[offset : offset + limit],
        "total_examples": len(rows),
        "offset": offset,
    }


@mcp.tool
def get_pilot_manifest() -> dict[str, Any]:
    """Return public metadata for the fixed, deterministic pilot input set."""
    return json.loads((PUBLIC / "pilot_manifest.json").read_text(encoding="utf-8"))


@mcp.tool
def get_pilot_input_batch(offset: int = 0, limit: int = 100) -> dict[str, Any]:
    """Return a bounded batch from the fixed deterministic pilot input set."""
    rows = _inputs("pilot_inputs.csv")
    offset, limit = max(offset, 0), min(max(limit, 1), 200)
    return {
        "examples": rows[offset : offset + limit],
        "total_examples": len(rows),
        "offset": offset,
    }


def main() -> None:
    mcp.run(show_banner=False)
