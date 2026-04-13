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


async def _solve_one(
    i: int,
    task: Task,
    config: MAWConfig,
    maw: MAW,
    scorer: ExactIntScorer,
    sem: asyncio.Semaphore,
    stage: str,
    total: int,
    progress: dict[str, int],
) -> TaskResult:
    async with sem:
        _log.info("[{}] Task {}/{} — solving...", stage, i + 1, total)
        try:
            run = await Controller(maw).run(task.input, config=config)
            if run.status == "error":
                err_msg = run.error.message if run.error else "unknown error"
                _log.warning(
                    "[{}] Task {}/{} pipeline error: {}", stage, i + 1, total, err_msg
                )
                scoring = None
                output = f"ERROR: {err_msg}"
            else:
                scoring = await scorer.evaluate(task, run.state)
                output = str(run.state.get("answer", ""))
        except Exception as exc:
            _log.warning("[{}] Task {}/{} failed: {}", stage, i + 1, total, exc)
            scoring = None
            output = f"ERROR: {exc}"

        is_correct = (scoring.score == 1.0) if scoring else False
        progress["done"] += 1
        if is_correct:
            progress["correct"] += 1
        _log.info(
            "[{}] Task {}/{} {} | expected={} running_acc={}/{} ({:.1%})",
            stage,
            i + 1,
            total,
            "✓" if is_correct else "✗",
            task.expected,
            progress["correct"],
            progress["done"],
            progress["correct"] / progress["done"],
        )

        return TaskResult(
            task_id=str(i),
            input=task.input[:200],
            expected=task.expected,
            output=output,
            score=scoring.score if scoring else 0.0,
            correct=is_correct,
        )


async def evaluate_on(
    config: MAWConfig,
    tasks: list[Task],
    maw: MAW,
    scorer: ExactIntScorer,
    stage: str = "eval",
    concurrency: int = 1,
) -> list[TaskResult]:
    total = len(tasks)
    sem = asyncio.Semaphore(max(1, concurrency))
    progress = {"done": 0, "correct": 0}
    _log.info(
        "[{}] Evaluating {} tasks with concurrency={}", stage, total, concurrency
    )
    coros = [
        _solve_one(i, t, config, maw, scorer, sem, stage, total, progress)
        for i, t in enumerate(tasks)
    ]
    return await asyncio.gather(*coros)


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
    trainset, valset, testset = load_math_dataset(
        seed=settings.seed,
        train_limit=settings.train_limit,
        val_limit=settings.val_limit,
        test_limit=settings.test_limit,
        test_repeats=settings.test_repeats,
    )
    seed_config = _build_seed_config(settings)

    opt_config = OptimizationConfig(
        seed=settings.seed,
        max_iterations=settings.max_iterations,
        patience=settings.patience,
        minibatch_size=settings.minibatch_size,
    )

    maw = MAW(worker_models=[settings.solver_model])
    scorer = ExactIntScorer()
    optimizer = Optimizer(maw, scorer=scorer, config=opt_config)
    opt_result = await optimizer.optimize(
        trainset, seed_config=seed_config, valset=valset
    )

    _log.info("Evaluating baseline on testset ({} tasks)", len(testset))
    baseline_tasks = await evaluate_on(
        seed_config, testset, maw, scorer,
        stage="baseline", concurrency=settings.concurrency,
    )

    if opt_result.best_config is seed_config or opt_result.best_config == seed_config:
        _log.info("Best config == seed — reusing baseline results for optimized eval")
        optimized_tasks = baseline_tasks
    else:
        _log.info("Evaluating optimized on testset ({} tasks)", len(testset))
        optimized_tasks = await evaluate_on(
            opt_result.best_config, testset, maw, scorer,
            stage="optimized", concurrency=settings.concurrency,
        )

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
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--test-repeats", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=None)
    args = parser.parse_args()

    overrides = {k: v for k, v in vars(args).items() if v is not None}
    overrides = {k.replace("-", "_"): v for k, v in overrides.items()}
    settings = AimeMathSettings(**overrides)

    asyncio.run(main(settings))
