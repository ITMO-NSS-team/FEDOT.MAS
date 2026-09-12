"""Private SAMPO retrieval diagnostics; never use GT for retrieval fitting."""

from __future__ import annotations

import csv
import json
import statistics
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sampo_baselines import (
    RankedPrediction,
    tfidf_char_ngrams_ranked,
    tfidf_char_word_hybrid_ranked,
    tfidf_word_ranked,
)
from sampo_evaluation import evaluate_predictions

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "artifacts" / "sampo_benchmark"
GT = ROOT / "artifacts" / "sampo_audit" / "private_ground_truth.csv"
OUT = ROOT / "artifacts" / "sampo_retrieval_diagnostics"
KS = (1, 3, 5, 10, 20, 50)
RANKING_K = max(KS)
METHODS: dict[str, Callable[[list[str], list[str], int], list[RankedPrediction]]] = {
    "char_tfidf": tfidf_char_ngrams_ranked,
    "word_tfidf": tfidf_word_ranked,
    "char_word_hybrid_rrf": tfidf_char_word_hybrid_ranked,
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def prediction_rows(
    examples: list[dict[str, str]], rankings: list[RankedPrediction]
) -> list[dict[str, str]]:
    return [
        {
            "example_id": example["example_id"],
            "top_1": ranking[0][0],
            "top_2": ranking[1][0],
            "top_3": ranking[2][0],
        }
        for example, ranking in zip(examples, rankings)
    ]


def recall_at(rankings: list[RankedPrediction], targets: list[str], k: int) -> float:
    return sum(target in [label for label, _ in ranking[:k]] for ranking, target in zip(rankings, targets)) / len(targets)


def save_rankings(
    subset: str, method: str, examples: list[dict[str, str]], rankings: list[RankedPrediction]
) -> None:
    path = OUT / f"{subset}_{method}_rankings_top_{RANKING_K}.csv"
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["example_id", "ranked_labels", "scores"])
        writer.writeheader()
        writer.writerows(
            {
                "example_id": example["example_id"],
                "ranked_labels": json.dumps([label for label, _ in ranking], ensure_ascii=False),
                "scores": json.dumps([score for _, score in ranking]),
            }
            for example, ranking in zip(examples, rankings)
        )


def bucket(value: int) -> str:
    if value == 1:
        return "1"
    if value <= 3:
        return "2-3"
    if value <= 10:
        return "4-10"
    return "11+"


def char_diagnostics(
    rankings: list[RankedPrediction], targets: list[str], support: Counter[str]
) -> dict[str, Any]:
    top1_scores = [ranking[0][1] for ranking in rankings]
    top2_scores = [ranking[1][1] for ranking in rankings]
    margins = [first - second for first, second in zip(top1_scores, top2_scores)]
    correct = [ranking[0][0] == target for ranking, target in zip(rankings, targets)]
    margin_ranges = (("<0.01", 0, 0.01), ("0.01-0.05", 0.01, 0.05), ("0.05-0.10", 0.05, 0.10), (">=0.10", 0.10, float("inf")))
    accuracy_by_margin = {}
    for name, low, high in margin_ranges:
        indexes = [index for index, margin in enumerate(margins) if low <= margin < high]
        accuracy_by_margin[name] = {
            "examples": len(indexes),
            "top_1_accuracy": sum(correct[index] for index in indexes) / len(indexes) if indexes else None,
        }
    support_buckets: dict[str, list[int]] = {}
    for index, target in enumerate(targets):
        support_buckets.setdefault(bucket(support[target]), []).append(index)
    by_support = {}
    for name, indexes in support_buckets.items():
        by_support[name] = {
            "examples": len(indexes),
            "top_1_accuracy": sum(correct[index] for index in indexes) / len(indexes),
            **{
                f"recall_at_{k}": sum(
                    targets[index] in [label for label, _ in rankings[index][:k]]
                    for index in indexes
                )
                / len(indexes)
                for k in KS
            },
        }
    confusions = Counter(
        (target, ranking[0][0])
        for ranking, target in zip(rankings, targets)
        if ranking[0][0] != target
    )
    return {
        "score_summary": {
            "top_1_mean": statistics.fmean(top1_scores),
            "top_1_median": statistics.median(top1_scores),
            "top_2_mean": statistics.fmean(top2_scores),
            "top_2_median": statistics.median(top2_scores),
            "top_1_top_2_margin_mean": statistics.fmean(margins),
            "top_1_top_2_margin_median": statistics.median(margins),
        },
        "accuracy_by_margin_bucket": accuracy_by_margin,
        "accuracy_and_recall_by_target_support_bucket": by_support,
        "most_common_top_1_confusion_pairs": [
            {"target_label": target, "predicted_label": predicted, "count": count}
            for (target, predicted), count in confusions.most_common(25)
        ],
    }


def run_subset(
    subset: str,
    examples: list[dict[str, str]],
    ground_truth: list[dict[str, str]],
    labels: list[str],
    support: Counter[str],
) -> dict[str, Any]:
    gt_by_id = {row["example_id"]: row for row in ground_truth}
    targets = [gt_by_id[row["example_id"]]["target_granular_name"] for row in examples]
    result: dict[str, Any] = {}
    for name, retrieve in METHODS.items():
        rankings = retrieve([row["raw_work_name"] for row in examples], labels, RANKING_K)
        predictions = prediction_rows(examples, rankings)
        # This validates IDs, labels, and rank uniqueness via the Phase 1 evaluator.
        top_three_metrics = evaluate_predictions(
            [gt_by_id[row["example_id"]] for row in examples], predictions, labels
        )
        recalls = {f"recall_at_{k}": recall_at(rankings, targets, k) for k in KS}
        result[name] = {**top_three_metrics, **recalls}
        save_rankings(subset, name, examples, rankings)
        if name == "char_tfidf":
            result[name]["diagnostics"] = char_diagnostics(rankings, targets, support)
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    full_inputs = read_rows(PUBLIC / "benchmark_inputs.csv")
    pilot_inputs = read_rows(PUBLIC / "pilot_inputs.csv")
    labels = [row["target_label"] for row in read_rows(PUBLIC / "allowed_target_labels.csv")]
    ground_truth = read_rows(GT)
    for index, row in enumerate(ground_truth, start=1):
        row["example_id"] = str(index)
    support = Counter(row["target_granular_name"] for row in ground_truth)
    results = {
        "retrieval_configuration": {
            "ranking_k": RANKING_K,
            "hybrid": "equal-weight reciprocal rank fusion, RRF constant 60",
            "private_gt_use": "evaluation and diagnostics only; never retrieval fitting",
        },
        "target_label_support_distribution": {
            name: sum(1 for count in support.values() if bucket(count) == name)
            for name in ("1", "2-3", "4-10", "11+")
        },
        "pilot_seed_42": run_subset("pilot", pilot_inputs, ground_truth, labels, support),
        "full": run_subset("full", full_inputs, ground_truth, labels, support),
    }
    full_methods = results["full"]
    best_name = max(full_methods, key=lambda name: full_methods[name]["recall_at_50"])
    best = full_methods[best_name]
    saturation_k = next(k for k in KS if best["recall_at_50"] - best[f"recall_at_{k}"] <= 0.02)
    top1_errors = 1 - best["recall_at_1"]
    error_recovered = {
        k: (best[f"recall_at_{k}"] - best["recall_at_1"]) / top1_errors
        if top1_errors
        else 0.0
        for k in (5, 10, 20)
    }
    bottleneck = "reranking" if best["recall_at_20"] >= 0.80 else "retrieval"
    results["answers"] = {
        "best_retrieval_method": best_name,
        "recall_saturation_k_within_2_points_of_recall_at_50": saturation_k,
        "top_1_errors_with_gold_in_top_k": {str(k): value for k, value in error_recovered.items()},
        "next_bottleneck": bottleneck,
        "future_semantic_reranker_candidate_set": max(10, saturation_k),
    }
    (OUT / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = [
        "# SAMPO retrieval diagnostics",
        "",
        "All retrievers fit only on public inputs and allowed labels. Private GT is used only after ranking.",
        "",
        "| Method | Pilot R@1 | Pilot R@3 | Pilot R@5 | Pilot R@10 | Pilot R@20 | Pilot R@50 | Full R@1 | Full R@3 | Full R@5 | Full R@10 | Full R@20 | Full R@50 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in METHODS:
        pilot, full = results["pilot_seed_42"][name], results["full"][name]
        report.append(
            f"| {name} | "
            + " | ".join(f"{pilot[f'recall_at_{k}']:.4f}" for k in KS)
            + " | "
            + " | ".join(f"{full[f'recall_at_{k}']:.4f}" for k in KS)
            + " |"
        )
    report.extend(
        [
            "",
            "## Answers",
            "",
            f"A. Best retrieval method: `{best_name}` by full recall@50.",
            f"B. Recall starts saturating at k={saturation_k}, defined as within 2 percentage points of recall@50.",
            "C. Of current best-method top-1 errors, gold is in top-5/top-10/top-20 for "
            + "/".join(f"{error_recovered[k]:.1%}" for k in (5, 10, 20))
            + ", respectively.",
            f"D. The next bottleneck is primarily `{bottleneck}` under the predeclared rule: recall@20 {'meets' if bottleneck == 'reranking' else 'does not meet'} 80%.",
            f"E. A future semantic reranker should receive {max(10, saturation_k)} candidates per example.",
            "",
            "## Char TF-IDF diagnostics (full benchmark)",
            "",
            "| Top-1 score mean | Top-2 score mean | Margin mean | Margin median |",
            "| ---: | ---: | ---: | ---: |",
            "| "
            + " | ".join(
                f"{full_methods['char_tfidf']['diagnostics']['score_summary'][key]:.4f}"
                for key in (
                    "top_1_mean",
                    "top_2_mean",
                    "top_1_top_2_margin_mean",
                    "top_1_top_2_margin_median",
                )
            )
            + " |",
            "",
            "| Margin bucket | Examples | Top-1 accuracy |",
            "| --- | ---: | ---: |",
            *[
                f"| {name} | {values['examples']} | {values['top_1_accuracy']:.4f} |"
                for name, values in full_methods["char_tfidf"]["diagnostics"][
                    "accuracy_by_margin_bucket"
                ].items()
            ],
            "",
            "Target-label support distribution (labels): "
            + ", ".join(
                f"{name}: {count}" for name, count in results["target_label_support_distribution"].items()
            )
            + ".",
            "",
            "Detailed support-bucket recall and the 25 most common top-1 confusion pairs are in `results.json`; ID-keyed top-50 rankings are saved alongside it.",
        ]
    )
    (OUT / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
