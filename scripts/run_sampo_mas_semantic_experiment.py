"""Run a regenerated public-only FEDOT.MAS SAMPO experiment."""

from __future__ import annotations

import asyncio
import csv
import json
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any

from fedotmas import MAS
from fedotmas._settings import get_meta_model
from fedotmas.plugins import LoggingPlugin, UnknownToolRecoveryPlugin
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins import BasePlugin

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "artifacts" / "sampo_benchmark"
OUT = ROOT / "artifacts" / "sampo_mas_semantic_experiment"
WORKER_MODELS = ["openai/gpt-5.6-luna", "openai/gpt-5-mini"]
TASK = """Map historical construction work names to one of the provided canonical
granular work labels. Design and execute an efficient workflow. The production
dataset contains millions of rows, so avoid unnecessary per-row LLM calls and
exploit batching/deduplication/code where useful. Maximize mapping accuracy:
lexical similarity alone is insufficient. Use LLM reasoning selectively while
avoiding one LLM call per row."""
PUBLIC_CONSTRAINTS = """Use only the public SAMPO benchmark inputs and the 466
allowed labels available through the provided tools. Do not access databases,
mapping tables, private ground truth, or labels not supplied by the tools.
Create a valid CSV with example_id, top_1, top_2, top_3; every prediction must
be an allowed label. For this run, use exactly the fixed public random pilot in
pilot_inputs.csv (seed 42; see get_pilot_manifest), produce exactly {count}
predictions for those IDs, and save them through the public workspace as
'{filename}'."""


class ModelCallCounter(BasePlugin):
    """Record model turns without affecting the generated MAS behavior."""

    def __init__(self) -> None:
        super().__init__(name="sampo_model_call_counter")
        self.calls: Counter[str] = Counter()

    async def after_model_callback(
        self, *, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> LlmResponse | None:
        self.calls[callback_context.agent_name] += 1
        return None

    def snapshot(self) -> dict[str, int]:
        return dict(sorted(self.calls.items()))


def _difference(current: dict[str, int], previous: dict[str, int]) -> dict[str, int]:
    return {
        name: count - previous.get(name, 0)
        for name, count in current.items()
        if count > previous.get(name, 0)
    }


def _query(count: int, filename: str) -> str:
    return f"{TASK}\n\n{PUBLIC_CONSTRAINTS.format(count=count, filename=filename)}"


def _prediction_rows(filename: str, expected: int) -> Path:
    path = PUBLIC / "mas_runs" / filename
    if not path.exists():
        raise RuntimeError(
            f"MAS did not create the required public prediction file: {path}"
        )
    with path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if len(rows) != expected:
        raise RuntimeError(
            f"Expected {expected} predictions in {path}, found {len(rows)}"
        )
    return path


def _previous_prediction_comparison() -> dict[str, Any]:
    previous = ROOT / "artifacts" / "sampo_mas_experiment" / "predictions.csv"
    baseline = PUBLIC / "predictions_tfidf_char_ngrams.csv"
    with previous.open(encoding="utf-8", newline="") as file:
        mas_rows = list(csv.DictReader(file))
    with baseline.open(encoding="utf-8", newline="") as file:
        baseline_rows = list(csv.DictReader(file))
    if len(mas_rows) != len(baseline_rows):
        raise RuntimeError(
            "Prior MAS and TF-IDF prediction files have different row counts"
        )
    columns = ("top_1", "top_2", "top_3")
    matching_tuples = sum(
        tuple(mas[column] for column in columns)
        == tuple(baseline[column] for column in columns)
        for mas, baseline in zip(mas_rows, baseline_rows)
    )
    return {
        "prior_mas_predictions": str(previous.relative_to(ROOT)),
        "tfidf_predictions": str(baseline.relative_to(ROOT)),
        "examples": len(mas_rows),
        "identical_top_3_tuples": matching_tuples,
        "identical_top_3_fraction": matching_tuples / len(mas_rows),
    }


def _pilot_metrics(predictions: Path) -> dict[str, float | int]:
    from evaluate_sampo_mas import GT
    from sampo_evaluation import evaluate_predictions

    with GT.open(encoding="utf-8", newline="") as file:
        ground_truth = list(csv.DictReader(file))
    for index, row in enumerate(ground_truth, start=1):
        row["example_id"] = str(index)
    with predictions.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    with (PUBLIC / "pilot_inputs.csv").open(encoding="utf-8", newline="") as file:
        pilot_ids = {row["example_id"] for row in csv.DictReader(file)}
    if {row["example_id"] for row in rows} != pilot_ids:
        raise RuntimeError("Pilot predictions do not contain exactly the manifest IDs")
    with (PUBLIC / "predictions_tfidf_char_ngrams_pilot.csv").open(
        encoding="utf-8", newline=""
    ) as file:
        tfidf_rows = list(csv.DictReader(file))
    if {row["example_id"] for row in tfidf_rows} != pilot_ids:
        raise RuntimeError("TF-IDF pilot predictions do not contain exactly the manifest IDs")
    labels = sorted({row["target_granular_name"] for row in ground_truth})
    values = evaluate_predictions(
        [row for row in ground_truth if row["example_id"] in pilot_ids], rows, labels
    )
    top3_columns = ("top_1", "top_2", "top_3")
    different_top_1 = sum(
        row["top_1"] != baseline["top_1"]
        for row, baseline in zip(
            sorted(rows, key=lambda item: int(item["example_id"])),
            sorted(tfidf_rows, key=lambda item: int(item["example_id"])),
        )
    )
    different_top_3 = sum(
        tuple(row[column] for column in top3_columns)
        != tuple(baseline[column] for column in top3_columns)
        for row, baseline in zip(
            sorted(rows, key=lambda item: int(item["example_id"])),
            sorted(tfidf_rows, key=lambda item: int(item["example_id"])),
        )
    )
    return {
        **values,
        "tfidf_different_top_1_fraction": different_top_1 / len(rows),
        "tfidf_different_top_3_tuple_fraction": different_top_3 / len(rows),
    }


async def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "prior_tfidf_comparison.json").write_text(
        json.dumps(_previous_prediction_comparison(), indent=2) + "\n", encoding="utf-8"
    )
    counter = ModelCallCounter()
    mas = MAS(
        meta_model=get_meta_model(),
        worker_models=WORKER_MODELS,
        mcp_servers=["sampo-python"],
        plugins=[LoggingPlugin(), UnknownToolRecoveryPlugin(), counter],
    )
    config = await mas.generate_config(_query(1000, "semantic_pilot_predictions.csv"))
    generated_calls = counter.snapshot()
    (OUT / "generated_mas_config.json").write_text(
        config.model_dump_json(indent=2), encoding="utf-8"
    )
    (OUT / "model_assignments.json").write_text(
        json.dumps(
            {
                "meta_model": get_meta_model(),
                "worker_model_candidates": WORKER_MODELS,
                "generated_config_agent_models": {
                    config.coordinator.name: config.coordinator.model,
                    **{worker.name: worker.model for worker in config.workers},
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    started = time.perf_counter()
    pilot_filename = "semantic_pilot_predictions.csv"
    # A failed prior pilot may have left this generated artifact behind. Do not
    # let a new run mistake it for output produced by its own MAS execution.
    (PUBLIC / "mas_runs" / pilot_filename).unlink(missing_ok=True)
    pilot_state = await mas.build_and_run(
        config, _query(1000, pilot_filename), timeout=1800
    )
    pilot_path = _prediction_rows(pilot_filename, 1000)
    pilot_calls = counter.snapshot()
    pilot_stats = {
        "prompt_tokens": mas.total_prompt_tokens,
        "completion_tokens": mas.total_completion_tokens,
        "llm_calls": _difference(pilot_calls, generated_calls),
    }
    pilot_metrics = _pilot_metrics(pilot_path)
    shutil.copyfile(pilot_path, OUT / "pilot_predictions.csv")
    (OUT / "pilot_state.json").write_text(
        json.dumps(pilot_state, default=str, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUT / "runtime_and_tokens.json").write_text(
        json.dumps(
            {
                "elapsed_seconds": time.perf_counter() - started,
                "pilot": pilot_stats,
                "topology_generation_llm_calls": generated_calls,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (OUT / "pilot_private_evaluation.json").write_text(
        json.dumps(pilot_metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    asyncio.run(run())
