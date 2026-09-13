"""Execute one saved Phase 4.1 configuration without regenerating it."""

from __future__ import annotations

import asyncio
import csv
import json
import shutil
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any

from fedotmas import MAS
from fedotmas.mas.models import MASConfig
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins import BasePlugin
from sampo_evaluation import evaluate_predictions

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "artifacts/sampo_phase_4_1/structural_review"
    / "batch_6e3973784dc64305b1fd051a90a00f35/config_01"
)
OUT = ROOT / "artifacts/sampo_phase_4_2" / "review_1_eab887787bf043e995852c6b9089c3b7"


class Telemetry(BasePlugin):
    """Record model-call observations without changing requests or responses."""

    def __init__(self) -> None:
        super().__init__(name="phase_4_2_telemetry")
        self._starts: dict[str, list[float]] = {}
        self.calls: list[dict[str, Any]] = []

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> None:
        del llm_request
        self._starts.setdefault(callback_context.agent_name, []).append(
            time.perf_counter()
        )

    async def after_model_callback(
        self, *, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> None:
        starts = self._starts.get(callback_context.agent_name, [])
        started = starts.pop() if starts else time.perf_counter()
        usage = llm_response.usage_metadata
        self.calls.append(
            {
                "call_index": len(self.calls) + 1,
                "agent": callback_context.agent_name,
                "prompt_tokens": (usage.prompt_token_count if usage else 0) or 0,
                "completion_tokens": (usage.candidates_token_count if usage else 0)
                or 0,
                "runtime_seconds": time.perf_counter() - started,
            }
        )


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _commit_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _evaluate(predictions: Path) -> dict[str, Any]:
    pilot = _rows(ROOT / "artifacts/sampo_benchmark/pilot_inputs.csv")
    truth = _rows(ROOT / "artifacts/sampo_audit/private_ground_truth.csv")
    for index, row in enumerate(truth, start=1):
        row["example_id"] = str(index)
    labels = [
        row["target_label"]
        for row in _rows(ROOT / "artifacts/sampo_benchmark/allowed_target_labels.csv")
    ]
    pilot_ids = {row["example_id"] for row in pilot}
    return evaluate_predictions(
        [row for row in truth if row["example_id"] in pilot_ids],
        _rows(predictions),
        labels,
    )


async def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"Refusing to overwrite existing run directory: {OUT}")
    config_path = SOURCE / "config.json"
    task_path = SOURCE / "task.txt"
    metadata_path = SOURCE / "metadata.json"
    generation_runtime_path = SOURCE.parent / "runtime_config.json"
    config = MASConfig.model_validate_json(config_path.read_text(encoding="utf-8"))
    task = task_path.read_text(encoding="utf-8")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    run_id = metadata["run_id"]
    if run_id not in task or run_id not in config.model_dump_json():
        raise RuntimeError("Saved configuration and task do not agree on the run_id")
    prediction_path = ROOT / "artifacts/sampo_benchmark/mas_runs" / f"{run_id}.csv"
    if prediction_path.exists():
        raise RuntimeError(
            f"Refusing to mix with an existing prediction artifact: {prediction_path}"
        )

    OUT.mkdir(parents=True)
    shutil.copy2(config_path, OUT / "config.json")
    shutil.copy2(task_path, OUT / "task.txt")
    shutil.copy2(metadata_path, OUT / "generation_metadata.json")
    shutil.copy2(generation_runtime_path, OUT / "generation_runtime_config.json")
    (OUT / "execution_metadata.json").write_text(
        json.dumps(
            {
                "execution_commit_sha": _commit_sha(),
                "source_config": str(SOURCE.relative_to(ROOT)),
                "run_id": run_id,
                "executed_once": True,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    telemetry = Telemetry()
    mas = MAS(mcp_servers=["sampo-benchmark", "sandbox-light"], plugins=[telemetry])
    started = time.perf_counter()
    try:
        final_state = await mas.build_and_run(config, task, timeout=1800)
        elapsed = time.perf_counter() - started
        (OUT / "final_state.json").write_text(
            json.dumps(final_state, default=str, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if not prediction_path.exists():
            raise RuntimeError(
                f"Execution did not finalize predictions: {prediction_path}"
            )
        shutil.copy2(prediction_path, OUT / "predictions.csv")
        (OUT / "evaluation.json").write_text(
            json.dumps(_evaluate(prediction_path), indent=2) + "\n", encoding="utf-8"
        )
        outcome: dict[str, Any] = {"status": "completed", "runtime_seconds": elapsed}
    except Exception as error:
        elapsed = time.perf_counter() - started
        (OUT / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        outcome = {
            "status": "failed",
            "runtime_seconds": elapsed,
            "error": f"{type(error).__name__}: {error}",
        }
        raise
    finally:
        (OUT / "telemetry.json").write_text(
            json.dumps(
                {
                    "llm_calls": telemetry.calls,
                    "total_prompt_tokens": sum(
                        call["prompt_tokens"] for call in telemetry.calls
                    ),
                    "total_completion_tokens": sum(
                        call["completion_tokens"] for call in telemetry.calls
                    ),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (OUT / "execution_log.json").write_text(
            json.dumps(outcome, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    asyncio.run(main())
