"""Private Phase 2.5 retrieval and fixed single-agent reranking diagnostic."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

from litellm import acompletion
from sampo_baselines import (
    bm25_token_ranked,
    tfidf_char_ngrams_ranked,
    tfidf_char_word_hybrid_ranked,
    tfidf_construction_token_ranked,
    tfidf_word_ranked,
)
from sampo_evaluation import evaluate_predictions

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "artifacts" / "sampo_benchmark"
GT = ROOT / "artifacts" / "sampo_audit" / "private_ground_truth.csv"
OUT = ROOT / "artifacts" / "sampo_phase_2_5"
MODEL = "openai/gpt-5.6-luna"
KS = (5, 10, 20, 50)
SUBSET_SIZE = 150
SUBSET_SEED = 2025
RETRIEVERS = {
    "char_tfidf": tfidf_char_ngrams_ranked,
    "word_tfidf": tfidf_word_ranked,
    "char_word_hybrid_rrf": tfidf_char_word_hybrid_ranked,
    "construction_token_tfidf": tfidf_construction_token_ranked,
    "bm25_construction_tokens": bm25_token_ranked,
}


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def recall(rankings, targets, k: int) -> float:
    return sum(target in [label for label, _ in ranking[:k]] for ranking, target in zip(rankings, targets)) / len(targets)


def prompt(raw: str, candidates: list[tuple[str, float]]) -> str:
    payload = [
        {"candidate_index": index, "label": label, "lexical_rank": index, "lexical_score": round(score, 8)}
        for index, (label, score) in enumerate(candidates, start=1)
    ]
    return (
        "Map the historical construction work name to the best supplied candidate label. "
        "Use only these candidates; do not invent or alter labels. Return JSON only: "
        '{"top_1_index": integer}.\n'
        f"raw_work_name: {json.dumps(raw, ensure_ascii=False)}\n"
        f"candidates: {json.dumps(payload, ensure_ascii=False)}"
    )


async def choose(raw: str, candidates: list[tuple[str, float]], semaphore: asyncio.Semaphore) -> tuple[str, dict[str, int], str]:
    async with semaphore:
        response = await acompletion(
            model=MODEL,
            messages=[{"role": "user", "content": prompt(raw, candidates)}],
            temperature=0,
            response_format={"type": "json_object"},
        )
    content = response.choices[0].message.content or ""
    match = re.search(r"\{.*\}", content, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object in response: {content[:200]!r}")
    index = int(json.loads(match.group())["top_1_index"])
    if not 1 <= index <= len(candidates):
        raise ValueError(f"Candidate index {index} is outside 1..{len(candidates)}")
    usage = getattr(response, "usage", None)
    tokens = {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
    }
    return candidates[index - 1][0], tokens, content


def metrics(examples, rankings, chosen, gt_by_id, labels) -> dict[str, Any]:
    targets = [gt_by_id[row["example_id"]]["target_granular_name"] for row in examples]
    lexical = [ranking[0][0] for ranking in rankings]
    prediction_rows = []
    for row, ranking, selected in zip(examples, rankings, chosen):
        fill = [label for label, _ in ranking if label != selected]
        prediction_rows.append({"example_id": row["example_id"], "top_1": selected, "top_2": fill[0], "top_3": fill[1]})
    evaluation = evaluate_predictions(
        [gt_by_id[row["example_id"]] for row in examples], prediction_rows, labels
    )
    oracle_indexes = [index for index, (target, ranking) in enumerate(zip(targets, rankings)) if target in [label for label, _ in ranking]]
    changed = Counter()
    for target, lexical_label, semantic_label in zip(targets, lexical, chosen):
        if lexical_label == semantic_label:
            changed["both_correct_unchanged" if lexical_label == target else "both_wrong_unchanged"] += 1
        elif lexical_label == target:
            changed["lexical_correct_semantic_wrong_harm"] += 1
        elif semantic_label == target:
            changed["lexical_wrong_semantic_correct_rescue"] += 1
        else:
            changed["both_wrong_different"] += 1
    lexical_oracle = sum(lexical[index] == targets[index] for index in oracle_indexes) / len(oracle_indexes) if oracle_indexes else 0.0
    semantic_oracle = sum(chosen[index] == targets[index] for index in oracle_indexes) / len(oracle_indexes) if oracle_indexes else 0.0
    return {
        "overall_top_1": evaluation["top_1_accuracy"],
        "oracle_retrievable_examples": len(oracle_indexes),
        "oracle_retrievable_top_1": semantic_oracle,
        "lexical_top_1_on_oracle_retrievable": lexical_oracle,
        "changed_prediction_analysis": dict(changed),
        "rescued": changed["lexical_wrong_semantic_correct_rescue"],
        "harmed": changed["lexical_correct_semantic_wrong_harm"],
        "net_gain": changed["lexical_wrong_semantic_correct_rescue"] - changed["lexical_correct_semantic_wrong_harm"],
    }


async def rerank(examples, rankings, gt_by_id, labels, name: str) -> dict[str, Any]:
    started = time.perf_counter()
    semaphore = asyncio.Semaphore(8)
    tasks = [choose(row["raw_work_name"], ranking, semaphore) for row, ranking in zip(examples, rankings)]
    outputs = await asyncio.gather(*tasks)
    chosen = [output[0] for output in outputs]
    token_totals = {key: sum(output[1][key] for output in outputs) for key in ("prompt_tokens", "completion_tokens")}
    result = metrics(examples, rankings, chosen, gt_by_id, labels)
    result.update({"model": MODEL, "calls": len(outputs), "runtime_seconds": time.perf_counter() - started, **token_totals})
    with (OUT / f"semantic_{name}_predictions.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["example_id", "lexical_top_1", "semantic_top_1"])
        writer.writeheader()
        writer.writerows({"example_id": row["example_id"], "lexical_top_1": ranking[0][0], "semantic_top_1": choice} for row, ranking, choice in zip(examples, rankings, chosen))
    return result


async def main(run_llm: bool) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pilot = rows(PUBLIC / "pilot_inputs.csv")
    labels = [row["target_label"] for row in rows(PUBLIC / "allowed_target_labels.csv")]
    ground_truth = rows(GT)
    for index, row in enumerate(ground_truth, start=1):
        row["example_id"] = str(index)
    gt_by_id = {row["example_id"]: row for row in ground_truth}
    targets = [gt_by_id[row["example_id"]]["target_granular_name"] for row in pilot]
    rankings_by_method = {name: retrieve([row["raw_work_name"] for row in pilot], labels, 50) for name, retrieve in RETRIEVERS.items()}
    retrieval = {name: {f"recall_at_{k}": recall(rankings, targets, k) for k in (1, 3, *KS)} for name, rankings in rankings_by_method.items()}
    best = "char_word_hybrid_rrf"
    hybrid = rankings_by_method[best]
    hybrid_by_id = {
        row["example_id"]: ranking for row, ranking in zip(pilot, hybrid)
    }
    hybrid_summary = {"top_1": retrieval[best]["recall_at_1"], **{f"recall_at_{k}": retrieval[best][f"recall_at_{k}"] for k in (3, *KS)}, "gap_top_1_to_recall_at_20": retrieval[best]["recall_at_20"] - retrieval[best]["recall_at_1"], "gap_top_1_to_recall_at_50": retrieval[best]["recall_at_50"] - retrieval[best]["recall_at_1"], "unrecoverable_at_50": 1 - retrieval[best]["recall_at_50"]}
    result: dict[str, Any] = {"best_hybrid_pilot_summary": hybrid_summary, "cheap_retrieval_recall": retrieval, "dense_retrieval": {"status": "not_run", "reason": "sentence-transformers/torch are not installed locally; no model was downloaded."}}
    subset = random.Random(SUBSET_SEED).sample(pilot, SUBSET_SIZE)
    (OUT / "semantic_subset_manifest.json").write_text(json.dumps({"seed": SUBSET_SEED, "examples": SUBSET_SIZE, "example_ids": [row["example_id"] for row in subset]}, indent=2) + "\n")
    if run_llm:
        result["semantic_small_subset"] = {}
        for k in KS:
            rankings = [hybrid_by_id[row["example_id"]][:k] for row in subset]
            result["semantic_small_subset"][f"k_{k}"] = await rerank(subset, rankings, gt_by_id, labels, f"subset_k_{k}")
        winner = max(KS, key=lambda k: (result["semantic_small_subset"][f"k_{k}"]["oracle_retrievable_top_1"], result["semantic_small_subset"][f"k_{k}"]["overall_top_1"], -k))
        result["semantic_full_pilot"] = {f"k_{winner}": await rerank(pilot, [ranking[:winner] for ranking in hybrid], gt_by_id, labels, f"pilot_k_{winner}")}
        result["selected_candidate_k"] = winner
    (OUT / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    report = ["# SAMPO Phase 2.5 diagnostic", "", "## Retrieval vs ranking", "", f"Hybrid top-1: {hybrid_summary['top_1']:.1%}; R@3/5/10/20/50: " + "/".join(f"{hybrid_summary[f'recall_at_{k}']:.1%}" for k in (3, 5, 10, 20, 50)) + ".", f"Top-1→R@20 gap: {hybrid_summary['gap_top_1_to_recall_at_20']:.1%}; top-1→R@50 gap: {hybrid_summary['gap_top_1_to_recall_at_50']:.1%}; unrecoverable at k=50: {hybrid_summary['unrecoverable_at_50']:.1%}.", "", "Retrieval failures are examples where gold is absent from top-k. Ranking failures are examples where gold is present but not lexical rank 1. Both are substantial here; neither failure class is inferred from the reranker prompt."]
    if run_llm:
        report.extend(["", "## Fixed single-agent semantic reranking", "", f"Model: `{MODEL}`. Same prompt and fixed subset for every k; only candidate count changes.", "", "| k | Overall top-1 | Oracle-retrievable top-1 | Lexical top-1 on oracle | Rescued | Harmed | Net gain | Calls |", "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
        for k in KS:
            value = result["semantic_small_subset"][f"k_{k}"]
            report.append(f"| {k} | {value['overall_top_1']:.4f} | {value['oracle_retrievable_top_1']:.4f} | {value['lexical_top_1_on_oracle_retrievable']:.4f} | {value['rescued']} | {value['harmed']} | {value['net_gain']} | {value['calls']} |")
        report.append(f"\nExpanded k={result['selected_candidate_k']} to the full fixed 1,000-example pilot. Full details, tokens, runtime, and changed-prediction classes are in `results.json`.")
    (OUT / "report.md").write_text("\n".join(report) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-llm", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(args.run_llm))
