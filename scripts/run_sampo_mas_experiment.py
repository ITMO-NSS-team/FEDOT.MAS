"""Generate and execute the first public-only FEDOT.MAS SAMPO experiment."""

from __future__ import annotations

import asyncio
import json
import shutil
import time
from pathlib import Path

from fedotmas import MAS
from fedotmas._settings import get_meta_model, get_worker_models

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "sampo_mas_experiment"
TASK = """Map historical construction work names to one of the provided canonical granular work labels. Design and execute an efficient workflow. Use only the public sampo-benchmark tools and safe code tools. Do not attempt to access databases, mapping tables, labels not supplied by the tool, or private ground truth. For this run, use the fixed random pilot (get_pilot_manifest and get_pilot_input_batch), produce its exact 1,000 IDs, and save predictions as '{filename}'. Do not run a full dataset experiment."""


async def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mas = MAS(mcp_servers=["sampo-benchmark", "sandbox-light"])
    config = await mas.generate_config(
        TASK.format(filename="pilot_predictions.csv")
    )
    (OUT / "generated_mas_config.json").write_text(
        config.model_dump_json(indent=2), encoding="utf-8"
    )
    (OUT / "model_assignments.json").write_text(
        json.dumps(
            {
                "meta_model": get_meta_model(),
                "worker_models": get_worker_models(),
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
    pilot_state = await mas.build_and_run(
        config, TASK.format(filename="pilot_predictions.csv"), timeout=900
    )
    predictions_dir = ROOT / "artifacts" / "sampo_benchmark" / "mas_runs"
    for source_name, destination_name in (
        ("pilot_predictions.csv", "pilot_predictions.csv"),
    ):
        source = predictions_dir / source_name
        if not source.exists():
            raise RuntimeError(
                f"Expected public prediction artifact was not created: {source}"
            )
        shutil.copyfile(source, OUT / destination_name)
    (OUT / "pilot_state.json").write_text(
        json.dumps(pilot_state, default=str, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUT / "runtime_and_tokens.json").write_text(
        json.dumps(
            {
                "elapsed_seconds": time.perf_counter() - started,
                "prompt_tokens": mas.total_prompt_tokens,
                "completion_tokens": mas.total_completion_tokens,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    asyncio.run(run())
