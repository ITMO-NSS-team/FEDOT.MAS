"""Private Phase 3 retrieval complementarity and candidate-union diagnostics."""

from __future__ import annotations

import csv
import json
from itertools import combinations
from pathlib import Path
from typing import Any

from sampo_baselines import (
    bm25_token_ranked,
    normalize,
    tfidf_char_ngrams_ranked,
    tfidf_char_word_hybrid_ranked,
    tfidf_construction_token_ranked,
    tfidf_word_ranked,
)

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "artifacts" / "sampo_benchmark"
GT = ROOT / "artifacts" / "sampo_audit" / "private_ground_truth.csv"
OUT = ROOT / "artifacts" / "sampo_phase_3"
KS = (5, 10, 20, 50)
METHODS = {
    "char_tfidf": tfidf_char_ngrams_ranked,
    "word_tfidf": tfidf_word_ranked,
    "hybrid": tfidf_char_word_hybrid_ranked,
    "bm25": bm25_token_ranked,
    "construction_token": tfidf_construction_token_ranked,
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def recall(candidates: list[list[str]], targets: list[str]) -> float:
    return sum(target in row for target, row in zip(targets, candidates)) / len(targets)


def rrf_union(rankings: dict[str, list[list[tuple[str, float]]]], k: int, source_depth: int | None = None) -> list[list[str]]:
    output = []
    for index in range(len(next(iter(rankings.values())))):
        scores: dict[str, float] = {}
        for ranking_set in rankings.values():
            for rank, (label, _) in enumerate(ranking_set[index][:source_depth], start=1):
                scores[label] = scores.get(label, 0.0) + 1 / (60 + rank)
        output.append([label for label, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]])
    return output


def raw_union(rankings: dict[str, list[list[tuple[str, float]]]], k: int) -> list[list[str]]:
    output = []
    for index in range(len(next(iter(rankings.values())))):
        labels: set[str] = set()
        for ranking_set in rankings.values():
            labels.update(label for label, _ in ranking_set[index][:k])
        output.append(sorted(labels))
    return output


def diverse_union(rankings: dict[str, list[list[tuple[str, float]]]], k: int) -> list[list[str]]:
    """Fixed diversity rule: avoid labels with >=90% normalized containment."""
    fused = rrf_union(rankings, 50, source_depth=20)
    output = []
    for candidates in fused:
        chosen = []
        for label in candidates:
            compact = normalize(label)
            if all(min(len(compact), len(normalize(old))) / max(len(compact), len(normalize(old))) < 0.9 for old in chosen):
                chosen.append(label)
            if len(chosen) == k:
                break
        output.append(chosen)
    return output


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = read_rows(PUBLIC / "pilot_inputs.csv")
    labels = [row["target_label"] for row in read_rows(PUBLIC / "allowed_target_labels.csv")]
    gt = read_rows(GT)
    targets_by_id = {str(index): row["target_granular_name"] for index, row in enumerate(gt, start=1)}
    targets = [targets_by_id[row["example_id"]] for row in inputs]
    rankings = {name: method([row["raw_work_name"] for row in inputs], labels, 50) for name, method in METHODS.items()}
    candidate_sets = {name: [[label for label, _ in ranking] for ranking in values] for name, values in rankings.items()}
    result: dict[str, Any] = {"methods": {}, "complementarity": {}, "candidate_unions": {}}
    for k in KS:
        recovered = {
            name: {
                index
                for index, rows in enumerate(candidate_sets[name])
                if targets[index] in rows[:k]
            }
            for name in METHODS
        }
        result["methods"][f"k_{k}"] = {name: len(indexes) / len(inputs) for name, indexes in recovered.items()}
        union = set().union(*recovered.values())
        result["complementarity"][f"k_{k}"] = {
            "raw_union_recall": len(union) / len(inputs),
            "raw_union_average_candidate_count": sum(len(row) for row in raw_union(rankings, k)) / len(inputs),
            "uniquely_recovered": {name: len(indexes - set().union(*(other for other_name, other in recovered.items() if other_name != name))) for name, indexes in recovered.items()},
            "pairwise_failure_overlap": {f"{left}__{right}": len((set(range(len(inputs))) - recovered[left]) & (set(range(len(inputs))) - recovered[right])) for left, right in combinations(METHODS, 2)},
        }
        variants = {
            "raw_union_budget": raw_union(rankings, max(1, k // len(METHODS))),
            "rrf_all_four": rrf_union({key: rankings[key] for key in ("char_tfidf", "word_tfidf", "bm25", "construction_token")}, k),
            "rrf_all_methods": rrf_union(rankings, k),
            "union_top_n_then_rrf": rrf_union(rankings, k, source_depth=k),
            "diverse_rrf": diverse_union(rankings, k),
        }
        result["candidate_unions"][f"k_{k}"] = {name: {"recall": recall(rows, targets), "average_candidate_count": sum(len(row) for row in rows) / len(rows)} for name, rows in variants.items()}
        for name, rows in variants.items():
            with (OUT / f"pilot_{name}_k_{k}.csv").open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=["example_id", "candidates"])
                writer.writeheader()
                writer.writerows({"example_id": row["example_id"], "candidates": json.dumps(candidates, ensure_ascii=False)} for row, candidates in zip(inputs, rows))
    (OUT / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = ["# SAMPO Phase 3", "", "## Retriever complementarity", "", "| k | Char | Word | Hybrid | BM25 | Construction | Raw union |", "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for k in KS:
        values = result["methods"][f"k_{k}"]
        report.append(f"| {k} | " + " | ".join(f"{values[name]:.4f}" for name in METHODS) + f" | {result['complementarity'][f'k_{k}']['raw_union_recall']:.4f} |")
    report.extend(["", "## Candidate unions", "", "| Candidate budget | Raw union | RRF four | RRF all | Union-top-N RRF | Diverse RRF |", "| ---: | ---: | ---: | ---: | ---: | ---: |"])
    for k in KS:
        values = result["candidate_unions"][f"k_{k}"]
        report.append(f"| {k} | " + " | ".join(f"{values[name]['recall']:.4f}" for name in values) + " |")
    report.extend(["", "Pairwise failure overlaps and uniquely recovered counts are in `results.json`. All candidate construction uses public inputs and labels only."])
    (OUT / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
