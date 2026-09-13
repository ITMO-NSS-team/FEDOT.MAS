"""Render the Phase 2.5 report from completed private diagnostic results."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "sampo_phase_2_5"
GT = ROOT / "artifacts" / "sampo_audit" / "private_ground_truth.csv"


def write_changed_prediction_classes() -> None:
    with GT.open(encoding="utf-8", newline="") as file:
        ground_truth = list(csv.DictReader(file))
    targets = {
        str(index): row["target_granular_name"]
        for index, row in enumerate(ground_truth, start=1)
    }
    for path in OUT.glob("semantic_*_predictions.csv"):
        with path.open(encoding="utf-8", newline="") as file:
            predictions = list(csv.DictReader(file))
        output = path.with_name(path.stem + "_changed_classes.csv")
        with output.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=["example_id", "lexical_top_1", "semantic_top_1", "private_class"],
            )
            writer.writeheader()
            for row in predictions:
                target = targets[row["example_id"]]
                lexical, semantic = row["lexical_top_1"], row["semantic_top_1"]
                if lexical == semantic:
                    classification = "both_correct_unchanged" if lexical == target else "both_wrong_unchanged"
                elif lexical == target:
                    classification = "lexical_correct_semantic_wrong_harm"
                elif semantic == target:
                    classification = "lexical_wrong_semantic_correct_rescue"
                else:
                    classification = "both_wrong_different"
                writer.writerow({**row, "private_class": classification})


def main() -> None:
    write_changed_prediction_classes()
    values = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
    hybrid = values["best_hybrid_pilot_summary"]
    small = values["semantic_small_subset"]
    selected_k = values["selected_candidate_k"]
    full = values["semantic_full_pilot"][f"k_{selected_k}"]
    lines = [
        "# SAMPO Phase 2.5 diagnostic",
        "",
        "## Retrieval vs ranking",
        "",
        f"Hybrid top-1: {hybrid['top_1']:.1%}; R@3/5/10/20/50: "
        + "/".join(f"{hybrid[f'recall_at_{k}']:.1%}" for k in (3, 5, 10, 20, 50))
        + ".",
        f"Top-1→R@20 gap: {hybrid['gap_top_1_to_recall_at_20']:.1%}; top-1→R@50 gap: {hybrid['gap_top_1_to_recall_at_50']:.1%}; unrecoverable at k=50: {hybrid['unrecoverable_at_50']:.1%}.",
        "",
        "Retrieval failures: gold absent from top-k. Ranking failures: gold present but not ranked first. At k=50, 20.0% are retrieval failures; the 51.4-point top-1→R@50 gap is the ranking-failure opportunity.",
        "",
        "## Cheap retrieval checks (pilot recall)",
        "",
        "| Method | R@10 | R@20 | R@50 |",
        "| --- | ---: | ---: | ---: |",
        *[
            f"| {name} | {metric['recall_at_10']:.4f} | {metric['recall_at_20']:.4f} | {metric['recall_at_50']:.4f} |"
            for name, metric in values["cheap_retrieval_recall"].items()
        ],
        "",
        "Dense retrieval was not run: no local sentence-transformers/torch model was available, and none was downloaded.",
        "",
        "## Fixed single-agent semantic reranking",
        "",
        "Model: `openai/gpt-5.6-luna`; same public-only prompt and fixed 150-example subset for each k.",
        "",
        "| k | Overall top-1 | Oracle top-1 | Lexical top-1 on oracle | Rescued | Harmed | Net gain | Calls |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for k in (5, 10, 20, 50):
        metric = small[f"k_{k}"]
        lines.append(f"| {k} | {metric['overall_top_1']:.4f} | {metric['oracle_retrievable_top_1']:.4f} | {metric['lexical_top_1_on_oracle_retrievable']:.4f} | {metric['rescued']} | {metric['harmed']} | {metric['net_gain']} | {metric['calls']} |")
    lines.extend(
        [
            "",
            f"Full fixed 1,000-example pilot at selected k={selected_k}: overall top-1 {full['overall_top_1']:.1%}; oracle-retrievable top-1 {full['oracle_retrievable_top_1']:.1%} vs lexical {full['lexical_top_1_on_oracle_retrievable']:.1%}; rescues/harm/net {full['rescued']}/{full['harmed']}/{full['net_gain']}; {full['calls']} calls, {full['prompt_tokens']} prompt tokens, {full['completion_tokens']} completion tokens, {full['runtime_seconds']:.1f}s.",
            "",
            "## Decision table",
            "",
            "| Case | Evidence | Decision |",
            "| --- | --- | --- |",
            "| A | Retrieval improves substantially and reranker helps | Build candidate retrieval MCP and proceed to generated MAS |",
            "| B | Retrieval improves but reranker does not | Do not expect MAS topology to solve it |",
            "| C — observed | R@50 remains 80.0%, while reranking improves oracle-retrievable accuracy by 6.3 points and nets +33 | Prioritize a retrieval candidate union before MAS |",
            "| D | Both retrieval and reranking are weak | Reconsider benchmark/public information |",
            "",
            "Changed-prediction classes, per-run tokens/runtime, and ID-keyed predictions are in `results.json` and `semantic_*_predictions.csv`.",
        ]
    )
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
