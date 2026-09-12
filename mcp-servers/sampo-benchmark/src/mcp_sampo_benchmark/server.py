from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

ROOT = Path(__file__).resolve().parents[4]
PUBLIC = ROOT / "artifacts" / "sampo_benchmark"
sys.path.insert(0, str(ROOT / "scripts"))
from sampo_baselines import tfidf_char_ngrams

mcp = FastMCP("sampo-benchmark")


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
    """Return public metadata for the fixed, deterministic SAMPO pilot."""
    import json

    return json.loads((PUBLIC / "pilot_manifest.json").read_text(encoding="utf-8"))


@mcp.tool
def get_pilot_input_batch(offset: int = 0, limit: int = 100) -> dict[str, Any]:
    """Return a bounded batch from the fixed 1,000-example random pilot."""
    rows = _inputs("pilot_inputs.csv")
    offset, limit = max(offset, 0), min(max(limit, 1), 200)
    return {
        "examples": rows[offset : offset + limit],
        "total_examples": len(rows),
        "offset": offset,
    }


@mcp.tool
def run_tfidf_baseline(
    max_examples: int = 1000,
    output_filename: str = "mas_predictions.csv",
    input_set: str = "benchmark",
) -> dict[str, Any]:
    """Run batched public TF-IDF character n-gram retrieval and save predictions only.

    This tool has no ground-truth/database access. max_examples is capped at the
    public dataset size; output contains example_id and top-3 allowed labels.
    """
    if input_set not in {"benchmark", "pilot"}:
        raise ValueError("input_set must be 'benchmark' or 'pilot'")
    rows, labels = _inputs("pilot_inputs.csv" if input_set == "pilot" else "benchmark_inputs.csv"), _labels()
    count = min(max(max_examples, 1), len(rows))
    predictions = tfidf_char_ngrams(
        [row["raw_work_name"] for row in rows[:count]], labels, 3
    )
    filename = Path(output_filename).name
    output = PUBLIC / "mas_runs" / filename
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=["example_id", "top_1", "top_2", "top_3"]
        )
        writer.writeheader()
        writer.writerows(
            {
                "example_id": row["example_id"],
                "top_1": prediction[0],
                "top_2": prediction[1],
                "top_3": prediction[2],
            }
            for row, prediction in zip(rows[:count], predictions)
        )
    return {
        "prediction_path": str(output.relative_to(ROOT)),
        "examples": count,
        "method": "tfidf_char_ngrams",
    }


def main() -> None:
    mcp.run(show_banner=False)
