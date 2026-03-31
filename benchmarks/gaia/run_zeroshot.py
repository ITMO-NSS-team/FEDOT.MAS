from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _utils import BenchmarkResult, TaskResult, save_result
from dataset import load_gaia_dataset
from fedotmas.common.logging import get_logger
from fedotmas.control._controller import Controller
from fedotmas.maw.maw import MAW
from fedotmas.optimize._state import Task
from scorer import GaiaScorer
from settings import GaiaSettings

_log = get_logger("bench.gaia.zeroshot")

TASK_DESCRIPTION = (
    "Answer general knowledge questions that may require reading files, "
    "web search, reasoning, and math. Provide a concise, exact answer."
)


async def evaluate_on(
    config,
    tasks: list[Task],
    maw: MAW,
    scorer: GaiaScorer,
) -> list[TaskResult]:
    results: list[TaskResult] = []
    for i, task in enumerate(tasks):
        try:
            run = await Controller(maw).run(task.input, config=config)
            scoring = await scorer.evaluate(task, run.state)
            output = str(run.state.get("answer", ""))
        except Exception as exc:
            _log.warning("Task {} failed: {}", i, exc)
            scoring = None
            output = f"ERROR: {exc}"

        results.append(
            TaskResult(
                task_id=str(i),
                input=task.input[:200],
                expected=task.expected,
                output=output,
                score=scoring.score if scoring else 0.0,
                correct=(scoring.score == 1.0) if scoring else False,
            )
        )
    return results


async def main(settings: GaiaSettings) -> BenchmarkResult:
    _, valset = load_gaia_dataset(
        difficulty=settings.difficulty,
        split=settings.split,
        seed=settings.seed,
    )
    _log.info("Loaded {} val tasks", len(valset))

    maw = MAW()
    config = await maw.generate_config(TASK_DESCRIPTION)
    _log.info("Generated config: {} agents", len(config.agents))

    scorer = GaiaScorer()
    _log.info("Evaluating zero-shot on valset ({} tasks)", len(valset))
    task_results = await evaluate_on(config, valset, maw, scorer)

    accuracy = (
        sum(t.correct for t in task_results) / len(task_results)
        if task_results
        else 0.0
    )

    result = BenchmarkResult(
        benchmark="gaia_zeroshot",
        metrics={"accuracy": accuracy},
        per_task=task_results,
    )

    path = save_result(result, Path(settings.output_dir))
    _log.info("GAIA Zero-shot accuracy: {:.1%}", accuracy)
    _log.info("Results saved to {}", path)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GAIA zero-shot benchmark")
    parser.add_argument("--difficulty", default=None, choices=["1", "2", "3", "all"])
    parser.add_argument("--split", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    overrides = {k: v for k, v in vars(args).items() if v is not None}
    overrides = {k.replace("-", "_"): v for k, v in overrides.items()}
    settings = GaiaSettings(**overrides)

    asyncio.run(main(settings))
