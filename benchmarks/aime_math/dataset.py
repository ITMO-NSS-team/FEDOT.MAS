from __future__ import annotations

import random

from datasets import load_dataset

from fedotmas.optimize._state import Task


def load_math_dataset(
    seed: int = 0,
    train_limit: int | None = None,
    val_limit: int | None = None,
    test_limit: int | None = None,
    test_repeats: int = 1,
) -> tuple[list[Task], list[Task], list[Task], dict[str, str]]:
    """Load AIME datasets from HuggingFace.

    Returns (trainset, valset, testset, solutions) where ``solutions``
    maps problem text → step-by-step solution (only available for
    train/val; test has none).

    * Train / val — AI-MO/aimo-validation-aime (AIME 2022-2024), 50/50 split.
    * Test — MathArena/aime_2025.
    """
    train_raw = load_dataset("AI-MO/aimo-validation-aime", "default", split="train")
    all_tasks: list[Task] = []
    solutions: dict[str, str] = {}
    for item in train_raw:
        problem = item["problem"]
        all_tasks.append(Task(input=problem, expected=str(item["answer"])))
        if item.get("solution"):
            solutions[problem] = item["solution"]

    random.Random(seed).shuffle(all_tasks)
    mid = len(all_tasks) // 2
    trainset = all_tasks[:mid]
    valset = all_tasks[mid:]

    if train_limit is not None:
        trainset = trainset[:train_limit]
    if val_limit is not None:
        valset = valset[:val_limit]

    test_raw = load_dataset("MathArena/aime_2025", "default", split="train")
    testset = [
        Task(input=item["problem"], expected=str(item["answer"])) for item in test_raw
    ]
    if test_limit is not None:
        testset = testset[:test_limit]
    if test_repeats > 1:
        testset = testset * test_repeats

    return trainset, valset, testset, solutions
