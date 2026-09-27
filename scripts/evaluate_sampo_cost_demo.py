#!/usr/bin/env python3
"""One-shot evaluator; only this program opens private ground truth."""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sampo_cost_demo import OUT, read_csv, paired_stats, end_to_end_accuracy
from sampo_cost_runtime import model_costs

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    finalized = OUT / "finalized.json"
    if finalized.exists():
        raise SystemExit("Evaluation already finalized; create a new experiment ID")
    manifest = json.loads((OUT / "manifest.json").read_text())
    ids = manifest["selected_ids"]
    gt = {r["example_id"]: r["target_granular_name"] for r in read_csv(OUT / "private_ground_truth.csv")}
    labels = {r["target_label"] for r in read_csv(OUT / "allowed_target_labels.csv")}
    predictions = {}
    telemetry = {}
    for system in ("tfidf", "cheap_single_agent", "fedotmas_cost_aware", "codex"):
        path = OUT / "runs" / args.run_id / system / "predictions.csv"
        if not path.exists(): raise SystemExit(f"Missing immutable prediction file: {path}")
        rows = read_csv(path)
        if len({r["example_id"] for r in rows}) != len(rows) or any(r["example_id"] not in ids for r in rows):
            raise SystemExit(f"Invalid IDs or duplicate predictions in {path}")
        for row in rows:
            vals = [row.get(k, "") for k in ("top_1", "top_2", "top_3")]
            if any(v not in labels for v in vals) or len(set(vals)) != 3: raise SystemExit(f"Invalid prediction schema in {path}")
        predictions[system] = {r["example_id"]: {**r, "completed": r.get("completed", "true").lower() == "true"} for r in rows}
        telemetry_path = path.parent / "telemetry.json"
        telemetry[system] = json.loads(telemetry_path.read_text()) if telemetry_path.exists() else {"model_calls": [], "tool_calls": 0, "runtime_seconds": 0}
    # Missing IDs represent execution failures and remain incorrect in the denominator.
    pricing = json.loads((OUT / "pricing.json").read_text())
    results = {}
    for system, pred in predictions.items():
        top1, conditional = end_to_end_accuracy(ids, gt, pred)
        top3 = sum(gt[i] in [pred.get(i, {}).get(k) for k in ("top_1", "top_2", "top_3")] for i in ids) / len(ids)
        calls = telemetry[system].get("model_calls", [])
        cost = model_costs(telemetry[system], pricing)
        correct = round(top1 * len(ids))
        results[system] = {"end_to_end_top1": top1, "top3": top3, "conditional_top1_on_completed": conditional, "total_inference_cost_usd": cost, "cost_complete":cost is not None, "cost_per_example_usd": cost / len(ids) if cost is not None else None, "cost_per_correct_top1_usd": cost / correct if cost is not None and correct else None, "cost_per_1000_examples_usd": cost * 1000 / len(ids) if cost is not None else None, "input_tokens": sum(c.get("input_tokens", 0) for c in calls), "output_tokens": sum(c.get("output_tokens", 0) for c in calls), "model_calls": len(calls), "tool_calls": telemetry[system].get("tool_calls", 0), "runtime_seconds": telemetry[system].get("runtime_seconds", 0), "completed_examples": sum(i in pred and pred[i].get("completed", False) for i in ids), "failure_rate": 1 - sum(i in pred and pred[i].get("completed", False) for i in ids) / len(ids), "llm_calls_per_example": len(calls) / len(ids)}
    comparisons = {other: paired_stats(ids, gt, predictions["fedotmas_cost_aware"], predictions[other]) for other in ("codex", "cheap_single_agent", "tfidf")}
    codex, fedot = results["codex"], results["fedotmas_cost_aware"]
    gap_pp = (codex["end_to_end_top1"] - fedot["end_to_end_top1"]) * 100
    reduction = codex["total_inference_cost_usd"] / fedot["total_inference_cost_usd"] if codex["cost_complete"] and fedot["cost_complete"] and fedot["total_inference_cost_usd"] else None
    report = {"experiment_id": args.run_id, "metrics": results, "paired_statistics": comparisons, "codex_minus_fedotmas_top1_gap_pp": gap_pp, "fedotmas_cost_reduction_factor_vs_codex": reduction, "cost_comparison_complete":codex["cost_complete"] and fedot["cost_complete"], "relative_accuracy_retained": fedot["end_to_end_top1"] / codex["end_to_end_top1"] if codex["end_to_end_top1"] else None, "preregistered_criterion_passed": gap_pp <= 3 and reduction is not None and reduction >= 2, "one_time_generation_cost": {"usd": 0, "status": "not configured"}}
    (OUT / "final_report.json").write_text(json.dumps(report, indent=2) + "\n")
    header = "| System | Top-1 | Top-3 | Cost | Cost / correct | Input tok. | Output tok. | Calls | Runtime | Completion |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    names = {"tfidf":"TF-IDF", "cheap_single_agent":"Cheap single agent", "fedotmas_cost_aware":"FEDOT.MAS cost-aware", "codex":"Codex"}
    lines = ["# SAMPO cost demo final report", "", header]
    for s, x in results.items():
        cost="N/A" if x["total_inference_cost_usd"] is None else f"${x['total_inference_cost_usd']:.6f}"
        per_correct="N/A" if x["cost_per_correct_top1_usd"] is None else f"${x['cost_per_correct_top1_usd']:.6f}"
        lines.append(f"| {names[s]} | {x['end_to_end_top1']:.3%} | {x['top3']:.3%} | {cost} | {per_correct} | {x['input_tokens']} | {x['output_tokens']} | {x['model_calls']} | {x['runtime_seconds']:.1f}s | {x['completed_examples']}/{len(ids)} |")
    lines += ["", f"Codex − FEDOT.MAS top-1 gap: {gap_pp:.2f} pp", f"FEDOT.MAS cost reduction factor vs Codex: {reduction}", f"Relative accuracy retained: {report['relative_accuracy_retained']}", f"Pre-registered criterion passed: {report['preregistered_criterion_passed']}", ""]
    (OUT / "final_report.md").write_text("\n".join(lines))
    finalized.write_text(json.dumps({"run_id": args.run_id, "prediction_files_immutable": True, "evaluation_complete": True}, indent=2) + "\n")

if __name__ == "__main__": main()
