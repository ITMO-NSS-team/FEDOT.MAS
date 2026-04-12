from __future__ import annotations

import random

from datasets import load_dataset

from fedotmas.optimize._state import Task


def load_math_dataset(
    seed: int = 0,
    train_limit: int | None = None,
    val_limit: int | None = None,
    test_limit: int | None = None,
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

    if train_limit:
        trainset = trainset[:train_limit]
    if val_limit:
        valset = valset[:val_limit]

    test_raw = load_dataset("MathArena/aime_2025", "default", split="train")
    testset = [
        Task(input=item["problem"], expected=str(item["answer"])) for item in test_raw
    ]
    if test_limit:
        testset = testset[:test_limit]

    return trainset, valset, testset
