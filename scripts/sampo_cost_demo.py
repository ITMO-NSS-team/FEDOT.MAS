"""GT-blind utilities for the isolated SAMPO cost demo."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
import unicodedata
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/sampo_cost_demo"
PREDICTION_COLUMNS = ("top_1", "top_2", "top_3")
PREREGISTRATION = {
    "primary_metrics": ["top1_accuracy", "top3_accuracy", "total_inference_cost", "cost_per_correct_top1"],
    "secondary_metrics": ["input_tokens", "output_tokens", "model_calls", "mcp_tool_calls", "wall_clock_seconds", "completed_examples", "failure_rate", "llm_calls_per_example", "cost_per_1000_examples", "conditional_top1_on_completed", "end_to_end_top1"],
    "success_criterion": {"fedotmas_top1_minimum_codex_top1_minus_pp": 3.0, "fedotmas_inference_cost_at_most_fraction_of_codex": 0.5},
    "relative_accuracy_retention": "FEDOTMAS_top1 / CODEX_top1; supplementary only; never substitutes absolute pp difference",
    "statistical_comparisons": ["fedotmas_cost_aware vs codex", "fedotmas_cost_aware vs cheap_single_agent", "fedotmas_cost_aware vs tfidf"],
    "evaluation_policy": "one-shot after all prediction files are immutable; failed/missing predictions count incorrect end-to-end",
    "selection_policy": "fixed seed; exclude every old pilot ID; exclude normalized work-name collisions with old pilot; no GT-based selection",
}


def normalized_name(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"[^\w]+", "", value, flags=re.UNICODE)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def construct_split(seed: int = 20260927, target: int = 1000) -> dict[str, Any]:
    """Create a one-time immutable, GT-separated split without label inspection."""
    OUT.mkdir(parents=True, exist_ok=True)
    immutable = [OUT / name for name in ("manifest.json", "public_inputs.csv", "allowed_target_labels.csv", "private_ground_truth.csv", "preregistration.json")]
    if any(path.exists() for path in immutable):
        raise FileExistsError("Frozen experiment already exists; use a new experiment directory/ID")
    public_path = ROOT / "artifacts/sampo_benchmark/benchmark_inputs.csv"
    pilot_path = ROOT / "artifacts/sampo_benchmark/pilot_inputs.csv"
    labels_path = ROOT / "artifacts/sampo_benchmark/allowed_target_labels.csv"
    gt_path = ROOT / "artifacts/sampo_audit/private_ground_truth.csv"
    source = read_csv(public_path)
    pilot = read_csv(pilot_path)
    labels = read_csv(labels_path)
    pilot_ids = {row["example_id"] for row in pilot}
    pilot_names = {normalized_name(row["raw_work_name"]) for row in pilot}
    eligible = [r for r in source if r["example_id"] not in pilot_ids and normalized_name(r["raw_work_name"]) not in pilot_names]
    if len(eligible) < target:
        raise ValueError(f"Only {len(eligible)} clean rows available; need {target}")
    selected = random.Random(seed).sample(eligible, target)
    selected_ids = {r["example_id"] for r in selected}
    # GT processing is confined to this preparatory process. It is copied only
    # to the private file and its path is never referenced by agent runners.
    gt_rows = read_csv(gt_path)
    gt_by_id = {str(i): row for i, row in enumerate(gt_rows, start=1)}
    if not selected_ids <= gt_by_id.keys():
        raise ValueError("Selected IDs do not resolve in private source mapping")
    private = [{"example_id": eid, "target_granular_name": gt_by_id[eid]["target_granular_name"]} for eid in sorted(selected_ids, key=int)]
    write_csv(OUT / "public_inputs.csv", ["example_id", "raw_work_name"], selected)
    write_csv(OUT / "allowed_target_labels.csv", ["target_label"], labels)
    write_csv(OUT / "private_ground_truth.csv", ["example_id", "target_granular_name"], private)
    (OUT / "preregistration.json").write_text(json.dumps(PREREGISTRATION, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "experiment_id": "sampo_cost_demo_v1", "frozen": True, "selection_seed": seed, "target_size": target,
        "selected_ids": [r["example_id"] for r in selected], "old_pilot_excluded_ids": sorted(pilot_ids, key=int),
        "exclusion_rules": ["all old pilot example IDs", "all normalized work_name values colliding with old pilot names", "NFKC + casefold + remove non-word characters"],
        "eligible_rows_before_sampling": len(eligible), "label_space_sha256": sha256(labels_path),
        "source_hashes": {str(p.relative_to(ROOT)): sha256(p) for p in (public_path, pilot_path, labels_path, gt_path)},
        "public_inputs_sha256": sha256(OUT / "public_inputs.csv"), "private_ground_truth_sha256": sha256(OUT / "private_ground_truth.csv"),
        "private_ground_truth_path_for_evaluator_only": "artifacts/sampo_cost_demo/private_ground_truth.csv",
        "agents_must_not_receive_private_ground_truth_path": True,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def call_cost(call: dict[str, Any], pricing: dict[str, Any]) -> float:
    if call.get("provider_cost_usd") is not None:
        return float(call["provider_cost_usd"])
    model = call["model"]
    rate = pricing["models"][model]
    uncached = max(0, int(call.get("input_tokens", 0)) - int(call.get("cached_input_tokens", 0)))
    return (uncached * rate["input_usd_per_1m"] + int(call.get("cached_input_tokens", 0)) * rate.get("cached_input_usd_per_1m", rate["input_usd_per_1m"]) + int(call.get("output_tokens", 0)) * rate["output_usd_per_1m"]) / 1_000_000


def end_to_end_accuracy(ids: list[str], gt: dict[str, str], predictions: dict[str, dict[str, str]]) -> tuple[float, float]:
    correct = sum(predictions.get(i, {}).get("top_1") == gt[i] for i in ids)
    completed = [i for i in ids if i in predictions and predictions[i].get("completed", True)]
    conditional = sum(predictions[i].get("top_1") == gt[i] for i in completed) / len(completed) if completed else 0.0
    return correct / len(ids), conditional


def paired_stats(ids: list[str], gt: dict[str, str], a: dict[str, dict[str, str]], b: dict[str, dict[str, str]], seed: int = 8128, samples: int = 10000) -> dict[str, Any]:
    wins_a = wins_b = both_correct = both_wrong = 0
    differences = []
    for i in ids:
        ca, cb = a.get(i, {}).get("top_1") == gt[i], b.get(i, {}).get("top_1") == gt[i]
        both_correct += ca and cb
        both_wrong += not ca and not cb
        wins_a += ca and not cb
        wins_b += cb and not ca
        differences.append(int(ca) - int(cb))
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(differences, k=len(differences))) / len(differences) for _ in range(samples))
    discordant = wins_a + wins_b
    # Exact two-sided binomial McNemar test, stable for the expected discordance range.
    tail = sum(math.comb(discordant, k) for k in range(min(wins_a, wins_b) + 1)) / (2 ** discordant) if discordant else 1.0
    return {"accuracy_difference_a_minus_b_pp": sum(differences) / len(differences) * 100, "paired_bootstrap_95_ci_pp": [means[int(samples * .025)] * 100, means[min(samples - 1, int(samples * .975))] * 100], "mcnemar_exact_two_sided_p": min(1.0, 2 * tail), "both_correct": both_correct, "both_wrong": both_wrong, "a_only_correct": wins_a, "b_only_correct": wins_b}
