"""Post-hoc oracle diagnostics for one completed Phase 5 run.

This reads finalized predictions, public retrieval artifacts, traces, and the
private audit truth. Output is written outside the run directory so official
reports remain immutable.
"""
from __future__ import annotations

import csv
import json
import os
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = os.environ.get("SAMPO_PHASE5_RUN_ID", "phase5_full_20260923_235200")
RUN_DIR = ROOT / "artifacts/sampo_phase_5" / f"full_{RUN}"
PREDICTIONS = ROOT / "artifacts/sampo_benchmark/mas_runs" / f"{RUN}.csv"
ARTIFACTS = ROOT / "artifacts/sampo_benchmark/candidate_artifacts"
OUT = ROOT / "artifacts/sampo_phase_5/posthoc_diagnostics" / RUN / "oracle_diagnostics.json"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def main() -> None:
    predictions = {row["example_id"]: row for row in read_csv(PREDICTIONS)}
    truth_names = {
        normalize(row["source_work_name"]): row["target_granular_name"]
        for row in read_csv(ROOT / "artifacts/sampo_audit/private_ground_truth.csv")
    }
    truth = {
        row["example_id"]: truth_names[normalize(row["raw_work_name"])]
        for row in read_csv(ROOT / "artifacts/sampo_benchmark/pilot_inputs.csv")
    }
    evidence_artifact: dict[str, str] = {}
    write_artifact: dict[str, str] = {}
    reviewed: set[str] = set()
    decisions: dict[str, list[int]] = {}
    artifact_ids: set[str] = set()
    for trace_path in sorted(RUN_DIR.glob("batch_*_trace.json")):
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        for call in trace.get("tool_calls", []):
            artifact_id = call.get("artifact_id")
            ids = call.get("ids") or []
            if not artifact_id or len(artifact_id) != 64:
                continue
            artifact_ids.add(artifact_id)
            if call.get("tool") == "get_candidate_evidence":
                for example_id in ids:
                    evidence_artifact[example_id] = artifact_id
                    reviewed.add(example_id)
            if call.get("tool") in {"save_candidate_predictions", "save_review_decisions"}:
                for example_id in ids:
                    write_artifact[example_id] = artifact_id
            if call.get("tool") == "save_review_decisions":
                for decision in call.get("decisions") or []:
                    decisions[decision["example_id"]] = decision["candidate_indices"]

    artifacts = {
        artifact_id: json.loads((ARTIFACTS / f"{artifact_id}.json").read_text(encoding="utf-8"))
        for artifact_id in artifact_ids
    }
    examples_by_artifact = {
        artifact_id: {example["example_id"]: example for example in artifact["examples"]}
        for artifact_id, artifact in artifacts.items()
    }

    rows: list[dict] = []
    for example_id, prediction in predictions.items():
        gold = truth[example_id]
        artifact_id = write_artifact.get(example_id) or evidence_artifact.get(example_id)
        if artifact_id is None:
            raise RuntimeError(f"No trace artifact found for finalized ID {example_id}")
        example = examples_by_artifact[artifact_id][example_id]
        methods = example["method_candidates"]
        union = set(methods)  # method_candidates is keyed by public label.
        fused_labels = [candidate["label"] for candidate in example["fused_candidates"]]
        final = [prediction[f"top_{position}"] for position in range(1, 4)]
        reviewed_example = example_id in reviewed
        fused_top3_hit = gold in fused_labels[:3]
        visible_top10_hit = gold in fused_labels[:10] if reviewed_example else None
        review_correction = review_regression = False
        if reviewed_example:
            baseline_top1 = fused_labels[0]
            review_correction = baseline_top1 != gold and prediction["top_1"] == gold
            review_regression = baseline_top1 == gold and prediction["top_1"] != gold
        rows.append({
            "example_id": example_id,
            "gold": gold,
            "final_top_1": prediction["top_1"],
            "final_top_3": final,
            "reviewed": reviewed_example,
            "gold_in_retriever_union": gold in union,
            "gold_in_fused_top_3": fused_top3_hit,
            "gold_in_visible_top_10": visible_top10_hit,
            "gold_in_final_top_1": prediction["top_1"] == gold,
            "gold_in_final_top_3": gold in final,
            "gold_in_fused_top_3_removed_from_final_top_3": fused_top3_hit and gold not in final,
            "gold_in_visible_top_10_absent_from_final_top_3": visible_top10_hit is True and gold not in final,
            "review_correction": review_correction,
            "review_regression": review_regression,
        })

    total = len(rows)
    reviewed_rows = [row for row in rows if row["reviewed"]]
    metric = lambda field, subset=rows: sum(bool(row[field]) for row in subset) / len(subset) if subset else None
    summary = {
        "run_id": RUN,
        "source_prediction_csv": str(PREDICTIONS.relative_to(ROOT)),
        "private_ground_truth_used_post_run_only": True,
        "examples": total,
        "reviewed_examples": len(reviewed_rows),
        "retriever_union_recall": metric("gold_in_retriever_union"),
        "fused_top_3_recall": metric("gold_in_fused_top_3"),
        "llm_visible_top_10_recall_on_reviewed": metric("gold_in_visible_top_10", reviewed_rows),
        "authoritative_final_top_1_accuracy": metric("gold_in_final_top_1"),
        "authoritative_final_top_3_accuracy": metric("gold_in_final_top_3"),
        "review_corrections": sum(row["review_correction"] for row in reviewed_rows),
        "review_regressions": sum(row["review_regression"] for row in reviewed_rows),
        "gold_in_fused_top_3_removed_from_final_top_3": sum(row["gold_in_fused_top_3_removed_from_final_top_3"] for row in rows),
        "gold_in_visible_top_10_absent_from_final_top_3": sum(row["gold_in_visible_top_10_absent_from_final_top_3"] for row in rows),
        "frozen_official_artifacts_modified": False,
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
