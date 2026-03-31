from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _utils import BenchmarkResult, CostSummary, TaskResult, save_result
from dataset import load_gaia_dataset
from fedotmas.common.logging import get_logger
from fedotmas.control._controller import Controller
from fedotmas.maw.maw import MAW
from fedotmas.optimize._config import OptimizationConfig
from fedotmas.optimize._optimizer import Optimizer
from fedotmas.optimize._state import Task
from scorer import GaiaScorer
from settings import GaiaSettings

_log = get_logger("bench.gaia.optimized")

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
    trainset, valset = load_gaia_dataset(
        difficulty=settings.difficulty,
        split=settings.split,
        seed=settings.seed,
    )
    _log.info("Loaded {} train, {} val tasks", len(trainset), len(valset))

    maw = MAW()

    _log.info("Generating initial config via meta-agent")
    seed_config = await maw.generate_config(TASK_DESCRIPTION)
    _log.info("Seed config: {} agents", len(seed_config.agents))

    opt_config = OptimizationConfig(
        seed=settings.seed,
        max_iterations=settings.max_iterations,
        patience=settings.patience,
        minibatch_size=settings.minibatch_size,
    )

    scorer = GaiaScorer()
    optimizer = Optimizer(maw, scorer=scorer, config=opt_config)
    opt_result = await optimizer.optimize(
        trainset, seed_config=seed_config, valset=valset
    )

    _log.info("Evaluating baseline on valset ({} tasks)", len(valset))
    baseline_tasks = await evaluate_on(seed_config, valset, maw, scorer)

    _log.info("Evaluating optimized on valset ({} tasks)", len(valset))
    optimized_tasks = await evaluate_on(opt_result.best_config, valset, maw, scorer)

    baseline_acc = (
        sum(t.correct for t in baseline_tasks) / len(baseline_tasks)
        if baseline_tasks
        else 0.0
    )
    optimized_acc = (
        sum(t.correct for t in optimized_tasks) / len(optimized_tasks)
        if optimized_tasks
        else 0.0
    )

    result = BenchmarkResult(
        benchmark="gaia_optimized",
        metrics={
            "baseline_accuracy": baseline_acc,
            "optimized_accuracy": optimized_acc,
            "improvement": optimized_acc - baseline_acc,
        },
        iterations=opt_result.iterations,
        cost=CostSummary(
            prompt_tokens=opt_result.total_prompt_tokens,
            completion_tokens=opt_result.total_completion_tokens,
            total_tokens=opt_result.total_prompt_tokens
            + opt_result.total_completion_tokens,
        ),
        per_task=optimized_tasks,
    )

    path = save_result(result, Path(settings.output_dir))
    _log.info("GAIA Optimized Results")
    _log.info("Baseline accuracy:   {:.1%}", baseline_acc)
    _log.info("Optimized accuracy:  {:.1%}", optimized_acc)
    _log.info("Improvement:         {:+.1%}", optimized_acc - baseline_acc)
    _log.info("Results saved to {}", path)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GAIA optimized benchmark")
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--difficulty", default=None, choices=["1", "2", "3", "all"])
    parser.add_argument("--split", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    overrides = {k: v for k, v in vars(args).items() if v is not None}
    overrides = {k.replace("-", "_"): v for k, v in overrides.items()}
    settings = GaiaSettings(**overrides)

    asyncio.run(main(settings))
