from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from fedotmas.common.logging import get_logger
from fedotmas.control._controller import Controller
from fedotmas.maw.maw import MAW
from fedotmas.maw.models import MAWAgentConfig, MAWConfig, MAWStepConfig
from fedotmas.optimize._config import OptimizationConfig
from fedotmas.optimize._optimizer import Optimizer
from fedotmas.optimize._state import Task

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _utils import BenchmarkResult, CostSummary, TaskResult, save_result
from config import INITIAL_PROMPT
from dataset import load_math_dataset
from scorer import ExactIntScorer
from settings import AimeMathSettings

_log = get_logger("fmbench.aime_math")


def _build_seed_config(settings: AimeMathSettings) -> MAWConfig:
    return MAWConfig(
        agents=[
            MAWAgentConfig(
                name="math_solver",
                instruction=INITIAL_PROMPT,
                model=settings.solver_model,
                output_key="answer",
            ),
        ],
        pipeline=MAWStepConfig(type="agent", agent_name="math_solver"),
    )


async def evaluate_on(
    config: MAWConfig,
    tasks: list[Task],
    maw: MAW,
    scorer: ExactIntScorer,
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


def report(result: BenchmarkResult) -> None:
    m = result.metrics
    _log.info("AIME Math Benchmark Results")
    _log.info("Iterations:          {}", result.iterations)
    _log.info("Baseline accuracy:   {:.1%}", m.get("baseline_accuracy", 0))
    _log.info("Optimized accuracy:  {:.1%}", m.get("optimized_accuracy", 0))
    _log.info("Improvement:         {:+.1%}", m.get("improvement", 0))
    if result.cost:
        _log.info("Total tokens:        {:,}", result.cost.total_tokens)


async def main(settings: AimeMathSettings) -> BenchmarkResult:
    trainset, valset, testset = load_math_dataset()
    seed_config = _build_seed_config(settings)

    opt_config = OptimizationConfig(
        seed=settings.seed,
        max_iterations=settings.max_iterations,
        patience=settings.patience,
        minibatch_size=settings.minibatch_size,
    )

    maw = MAW()
    scorer = ExactIntScorer()
    optimizer = Optimizer(maw, scorer=scorer, config=opt_config)
    opt_result = await optimizer.optimize(
        trainset, seed_config=seed_config, valset=valset
    )

    _log.info("Evaluating baseline on testset ({} tasks)", len(testset))
    baseline_tasks = await evaluate_on(seed_config, testset, maw, scorer)

    _log.info("Evaluating optimized on testset ({} tasks)", len(testset))
    optimized_tasks = await evaluate_on(opt_result.best_config, testset, maw, scorer)

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
        benchmark="aime_math",
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
    report(result)
    _log.info("Results saved to {}", path)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AIME Math benchmark")
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    overrides = {k: v for k, v in vars(args).items() if v is not None}
    overrides = {k.replace("-", "_"): v for k, v in overrides.items()}
    settings = AimeMathSettings(**overrides)

    asyncio.run(main(settings))
