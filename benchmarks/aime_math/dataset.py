from __future__ import annotations

import random

from datasets import load_dataset

from fedotmas.optimize._state import Task


def load_math_dataset(
    seed: int = 0,
) -> tuple[list[Task], list[Task], list[Task]]:
    """Load AIME datasets from HuggingFace.

    Returns (trainset, valset, testset) as lists of ``Task``.

    * Train / val — AI-MO/aimo-validation-aime (AIME 2022-2024), 50/50 split.
    * Test — MathArena/aime_2025.
    """
    train_raw = load_dataset("AI-MO/aimo-validation-aime", "default", split="train")
    all_tasks: list[Task] = []
    for item in train_raw:
        all_tasks.append(Task(input=item["problem"], expected=str(item["answer"])))

    random.Random(seed).shuffle(all_tasks)
    mid = len(all_tasks) // 2
    trainset = all_tasks[:mid]
    valset = all_tasks[mid:]

    test_raw = load_dataset("MathArena/aime_2025", "default", split="train")
    testset = [
        Task(input=item["problem"], expected=str(item["answer"])) for item in test_raw
    ]

    return trainset, valset, testset
