from __future__ import annotations

import os
import random
from pathlib import Path

import huggingface_hub
from datasets import load_dataset
from fedotmas.optimize._state import Task

HF_TOKEN = os.getenv("HF_TOKEN")
DATASET_CACHE_DIR = Path(__file__).parent / "gaia_data"


def load_gaia_dataset(
    difficulty: str = "all",
    split: str = "validation",
    seed: int = 0,
    train_ratio: float = 0.5,
) -> tuple[list[Task], list[Task]]:
    """Load GAIA dataset from HuggingFace.

    Returns (trainset, valset) as lists of ``Task``.

    GAIA validation split is used for both train/val since
    the test split has no ground truth labels.
    """
    config = "2023_all" if difficulty == "all" else f"2023_level{difficulty}"

    raw = load_dataset(
        "gaia-benchmark/GAIA",
        config,
        split=split,
        trust_remote_code=True,
        token=HF_TOKEN,
    )

    huggingface_hub.snapshot_download(
        "gaia-benchmark/GAIA",
        repo_type="dataset",
        local_dir=str(DATASET_CACHE_DIR),
        token=HF_TOKEN,
    )

    tasks: list[Task] = []
    for item in raw:
        file_path = item["file_path"]
        if file_path:
            file_path = str((DATASET_CACHE_DIR / file_path).resolve())

        question = item["Question"]
        if file_path:
            question = f"File path: {file_path}\n{question}"

        tasks.append(Task(input=question, expected=item["Final answer"]))

    random.Random(seed).shuffle(tasks)
    mid = int(len(tasks) * train_ratio)
    return tasks[:mid], tasks[mid:]
