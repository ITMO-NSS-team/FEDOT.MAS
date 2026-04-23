from __future__ import annotations

import random

from datasets import load_dataset

from fedotmas.optimize._state import Task

# Default split sizes drawn from HotpotQA's `validation` (the only HF split with
# gold answers). 50/50/100 keeps the per-iteration eval cheap while staying
# comparable to GEPA paper's HotpotQA setup (~150 train / 300 val).
_DEFAULT_TRAIN = 50
_DEFAULT_VAL = 50
_DEFAULT_TEST = 100


def _format_context(titles: list[str], sentences: list[list[str]]) -> str:
    blocks: list[str] = []
    for i, (title, sents) in enumerate(zip(titles, sentences), start=1):
        body = " ".join(sents).strip()
        blocks.append(f"[{i}] Title: {title}\n{body}")
    return "\n\n".join(blocks)


def _build_input(question: str, context: dict) -> str:
    ctx = _format_context(context["title"], context["sentences"])
    return f"{ctx}\n\nQuestion: {question}"


def load_hotpot_dataset(
    seed: int = 0,
    train_limit: int | None = None,
    val_limit: int | None = None,
    test_limit: int | None = None,
    test_repeats: int = 1,
) -> tuple[list[Task], list[Task], list[Task]]:
    """Load HotpotQA distractor setting from HuggingFace.

    The HF `validation` split (~7k items) has gold answers, so we shuffle it
    and slice into train / val / test. Each task input bundles the 10
    distractor paragraphs with the question; the expected output is the
    short answer string.
    """
    raw = load_dataset(
        "hotpot_qa", "distractor", split="validation", trust_remote_code=True
    )
    all_tasks: list[Task] = []
    for item in raw:
        text = _build_input(item["question"], item["context"])
        answer = str(item["answer"]).strip()
        all_tasks.append(Task(input=text, expected=answer))

    random.Random(seed).shuffle(all_tasks)

    a = _DEFAULT_TRAIN
    b = a + _DEFAULT_VAL
    c = b + _DEFAULT_TEST
    trainset = all_tasks[:a]
    valset = all_tasks[a:b]
    testset = all_tasks[b:c]

    if train_limit:
        trainset = trainset[:train_limit]
    if val_limit:
        valset = valset[:val_limit]
    if test_limit:
        testset = testset[:test_limit]
    if test_repeats > 1:
        testset = testset * test_repeats

    return trainset, valset, testset
