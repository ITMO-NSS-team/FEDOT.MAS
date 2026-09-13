"""Generate five Phase 4.1 MAS configurations for structural review only."""

from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
import uuid
from pathlib import Path

from fedotmas import MAS
from fedotmas._settings import get_meta_model

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "sampo_phase_4_1" / "structural_review"
RUNTIME_FILES = (
    ROOT
    / "mcp-servers"
    / "sampo-benchmark"
    / "src"
    / "mcp_sampo_benchmark"
    / "server.py",
    ROOT / "scripts" / "sampo_baselines.py",
    ROOT / "packages" / "fedotmas" / "src" / "fedotmas" / "meta" / "mas_prompts.py",
)
TASK = """Map each public historical work name in the fixed pilot to one of the
allowed labels. Maximize accuracy while using computation efficiently. Public
inputs, independent candidate retrieval, safe persistent prediction storage,
and Python/data processing are available. Use bounded batches and deterministic
computation where useful. Use semantic reasoning only when it can improve an
uncertain decision; do not move large datasets through agent messages. Use only
public inputs and labels. Produce valid ranked predictions for every pilot ID."""


def _commit_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _runtime_config() -> dict[str, object]:
    files = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in RUNTIME_FILES
    }
    return {
        "git_commit": _commit_sha(),
        "mcp_servers": ["sampo-benchmark", "sandbox-light"],
        "candidate_retrieval": {
            "tool": "retrieve_candidates",
            "method_catalog_tool": "list_methods",
            "methods": [
                "bm25_token",
                "char_tfidf",
                "char_word_fusion",
                "construction_token_tfidf",
                "word_tfidf",
            ],
        },
        "prediction_storage_tools": [
            "save_prediction_batch",
            "replace_prediction_batch",
            "import_prediction_file",
            "finalize_predictions",
        ],
        "pilot_input_tools": ["get_pilot_manifest", "get_pilot_input_batch"],
        "runtime_file_sha256": files,
    }


async def _generate(index: int, batch_dir: Path) -> None:
    run_id = f"review_{index}_{uuid.uuid4().hex}"
    task = f"{TASK}\nUse prediction storage run_id {run_id} if persistence is needed."
    mas = MAS(
        meta_model=get_meta_model(),
        worker_models=["openai/gpt-5.6-luna", "openai/gpt-5-mini"],
        mcp_servers=["sampo-benchmark", "sandbox-light"],
    )
    config = await mas.generate_config(task)
    config_dir = batch_dir / f"config_{index:02d}"
    config_dir.mkdir()
    (config_dir / "config.json").write_text(config.model_dump_json(indent=2) + "\n")
    (config_dir / "task.txt").write_text(task + "\n")
    (config_dir / "metadata.json").write_text(
        json.dumps(
            {
                "index": index,
                "run_id": run_id,
                "meta_model": get_meta_model(),
                "workers": {worker.name: worker.model for worker in config.workers},
                "executed": False,
            },
            indent=2,
        )
        + "\n"
    )


async def main() -> None:
    batch_dir = OUT / f"batch_{uuid.uuid4().hex}"
    batch_dir.mkdir(parents=True)
    (batch_dir / "runtime_config.json").write_text(
        json.dumps(_runtime_config(), indent=2) + "\n"
    )
    await asyncio.gather(*(_generate(index, batch_dir) for index in range(1, 6)))
    (batch_dir / "README.md").write_text(
        "# Phase 4.1 structural review\n\n"
        "This batch contains exactly five generated configurations. None was executed.\n"
    )


if __name__ == "__main__":
    asyncio.run(main())
