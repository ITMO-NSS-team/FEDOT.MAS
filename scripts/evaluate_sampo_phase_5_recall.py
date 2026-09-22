"""Decompose finalized Phase 5 errors into retrieval and ranking failures."""
from __future__ import annotations

import csv
import json
import os
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = os.environ.get("SAMPO_PHASE5_RUN_ID", "phase5_full_c14f4c3f1f4343ec90c4662a845439c7_config01_schema")
RUN_DIR = ROOT / "artifacts/sampo_phase_5" / f"full_{RUN}"
ARTIFACTS = ROOT / "artifacts/sampo_benchmark/candidate_artifacts"
OUT = RUN_DIR / "recall_decomposition.json"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def normalize_name(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def main() -> None:
    predictions = {row["example_id"]: row for row in rows(ROOT / f"artifacts/sampo_benchmark/mas_runs/{RUN}.csv")}
    truth_rows = rows(ROOT / "artifacts/sampo_audit/private_ground_truth.csv")
    truth_by_name = {normalize_name(row["source_work_name"]): row["target_granular_name"] for row in truth_rows}
    pilot_rows = rows(ROOT / "artifacts/sampo_benchmark/pilot_inputs.csv")
    truth = {row["example_id"]: truth_by_name[normalize_name(row["raw_work_name"])] for row in pilot_rows}
    seen: dict[str, set[str]] = {}
    final_artifacts: dict[str, set[str]] = {}
    for trace_path in sorted(RUN_DIR.glob("batch_*_trace.json")):
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        for call in trace.get("tool_calls", []):
            artifact_id = call.get("artifact_id")
            if not artifact_id or len(artifact_id) != 64:
                continue
            ids = set(call.get("ids") or [])
            if call.get("tool") == "get_candidate_evidence":
                for example_id in ids:
                    seen.setdefault(example_id, set()).add(artifact_id)
            if call.get("tool") in {"save_candidate_predictions", "save_review_decisions"}:
                for example_id in ids:
                    final_artifacts.setdefault(example_id, set()).add(artifact_id)
    artifacts: dict[str, dict] = {}
    for artifact_id in {item for values in seen.values() for item in values} | {
        item for values in final_artifacts.values() for item in values
    }:
        artifacts[artifact_id] = json.loads((ARTIFACTS / f"{artifact_id}.json").read_text(encoding="utf-8"))

    categories = Counter()
    rows_out = []
    for example_id, prediction in predictions.items():
        gold = truth[example_id]
        observed_artifacts = [artifacts[a] for a in seen.get(example_id, set()) | final_artifacts.get(example_id, set())]
        write_artifacts = [artifacts[a] for a in final_artifacts.get(example_id, set())]
        candidate_union = {
            candidate["label"]
            for artifact in observed_artifacts
            for example in artifact["examples"]
            if example["example_id"] == example_id
            for candidate in example["fused_candidates"]
        }
        write_top3 = {
            candidate["label"]
            for artifact in write_artifacts
            for example in artifact["examples"]
            if example["example_id"] == example_id
            for candidate in example["fused_candidates"][:3]
        }
        final_correct = prediction["top_1"] == gold
        # Final correctness is authoritative for the prediction file.  Trace
        # evidence can be truncated at process shutdown, so do not relabel a
        # correct final prediction as a retrieval miss when its evidence record
        # is incomplete.
        if final_correct:
            category = "final_top1_correct"
        elif gold not in candidate_union:
            category = "gold_absent_from_retrieved_candidates"
        elif gold not in write_top3:
            category = "gold_retrieved_but_outside_fused_top3"
        elif not final_correct:
            category = "gold_in_fused_top3_but_final_top1_wrong"
        else:
            category = "final_top1_correct"
        categories[category] += 1
        rows_out.append({
            "example_id": example_id,
            "gold": gold,
            "final_top1": prediction["top_1"],
            "retrieved_candidate_count": len(candidate_union),
            "gold_in_retrieved_candidates": gold in candidate_union,
            "gold_in_write_fused_top3": gold in write_top3,
            "category": category,
        })
    result = {
        "run_id": RUN,
        "private_ground_truth_used_post_run_only": True,
        "examples": len(rows_out),
        "categories": dict(categories),
        "candidate_union_recall": sum(row["gold_in_retrieved_candidates"] for row in rows_out) / len(rows_out),
        "write_fused_top3_recall": sum(row["gold_in_write_fused_top3"] for row in rows_out) / len(rows_out),
        "final_top1_accuracy": sum(predictions[row["example_id"]]["top_1"] == row["gold"] for row in rows_out) / len(rows_out),
        "rows": rows_out,
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
