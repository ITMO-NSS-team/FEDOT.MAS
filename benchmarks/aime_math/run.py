from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from fedotmas.common.logging import get_logger
from fedotmas.control._controller import Controller
from fedotmas.maw.maw import MAW
from fedotmas.maw.models import MAWConfig, MAWStepConfig
from fedotmas.optimize._config import OptimizationConfig
from fedotmas.optimize._optimizer import Optimizer
from fedotmas.optimize._state import Task

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _utils import BenchmarkResult, CostSummary, TaskResult, save_result
from dataset import load_math_dataset
from scorer import ExactIntScorer
from settings import AimeMathSettings

_log = get_logger("fmbench.aime_math")

_SEED_CONFIG_PATH = Path(__file__).parent / "seed_config.json"


def _load_seed_config(settings: AimeMathSettings) -> MAWConfig:
    if not _SEED_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Seed config not found at {_SEED_CONFIG_PATH}. "
            f"Run `python benchmarks/aime_math/generate_seed.py` first."
        )
    _log.info("Loading seed config from {}", _SEED_CONFIG_PATH)
    config = MAWConfig.model_validate_json(_SEED_CONFIG_PATH.read_text())
    if settings.max_output_tokens is not None:
        for agent in config.agents:
            agent.max_output_tokens = settings.max_output_tokens
    return config


def _final_output_key(config: MAWConfig) -> str:
    """Walk the pipeline tree to find the output_key of the last-executing agent."""
    def _last_agent_name(node: MAWStepConfig) -> str | None:
        if node.type == "agent":
            return node.agent_name
        if node.children:
            return _last_agent_name(node.children[-1])
        return None

    name = _last_agent_name(config.pipeline)
    if name is None:
        raise ValueError("Could not locate final agent in pipeline")
    agent = next(a for a in config.agents if a.name == name)
    return agent.output_key


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
                output = str(run.state.get(_final_output_key(config), ""))
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
    if "train_accuracy" in m:
        _log.info("Train accuracy:      {:.1%} (best on full trainset)", m["train_accuracy"])
        _log.info("Val accuracy:        {:.1%} (best on full valset)", m.get("val_accuracy", 0))
    if result.cost:
        _log.info("Total tokens:        {:,}", result.cost.total_tokens)


async def main(settings: AimeMathSettings) -> BenchmarkResult:
    trainset, valset, testset, solutions = load_math_dataset(
        seed=settings.seed,
        train_limit=settings.train_limit,
        val_limit=settings.val_limit,
        test_limit=settings.test_limit,
        test_repeats=settings.test_repeats,
    )
    seed_config = _load_seed_config(settings)
    answer_key = _final_output_key(seed_config)
    _log.info("Final agent output_key detected as {!r}", answer_key)

    opt_config = OptimizationConfig(
        seed=settings.seed,
        max_iterations=settings.max_iterations,
        max_evaluations=settings.max_evaluations,
        patience=settings.patience,
        minibatch_size=settings.minibatch_size,
        use_merge=settings.use_merge,
        eval_concurrency=settings.eval_concurrency,
        checkpoint_path=settings.checkpoint_path,
    )

    maw = MAW(worker_models=[settings.solver_model])
    scorer = ExactIntScorer(output_key=answer_key, solutions=solutions)

    train_eval_tasks: list[TaskResult] = []

    if settings.max_iterations == 0:
        _log.info("max_iterations=0 — skipping optimizer, evaluating seed on testset only")
        opt_result = None
        baseline_tasks = await evaluate_on(
            seed_config, testset, maw, scorer,
            stage="baseline", concurrency=settings.concurrency,
        )
        optimized_tasks = baseline_tasks
    else:
        optimizer = Optimizer(maw, scorer=scorer, config=opt_config)
        opt_result = await optimizer.optimize(
            trainset, seed_config=seed_config, valset=valset
        )

        if settings.skip_baseline_test:
            _log.info("Skipping baseline test evaluation (--skip-baseline-test)")
            baseline_tasks = []
        else:
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

        if settings.eval_best_on_train:
            # Diagnostic: full-train eval of best_config to surface a train/val/test gap.
            # Val accuracy is already known (best_score = mean over full valset, since
            # every accepted candidate gets a full-val eval), so we only need fresh train.
            # Train ≈ val ≪ test → distribution shift (train/val one set, test another).
            # Train < val → selection bias on val (instruction overfit to val-fold).
            _log.info("Evaluating best on full trainset ({} tasks)", len(trainset))
            train_eval_tasks = await evaluate_on(
                opt_result.best_config, trainset, maw, scorer,
                stage="train-final", concurrency=settings.concurrency,
            )

    if baseline_tasks:
        baseline_acc = sum(t.correct for t in baseline_tasks) / len(baseline_tasks)
    elif settings.baseline_accuracy is not None:
        baseline_acc = settings.baseline_accuracy
    else:
        baseline_acc = 0.0
    optimized_acc = (
        sum(t.correct for t in optimized_tasks) / len(optimized_tasks)
        if optimized_tasks
        else 0.0
    )

    if opt_result is not None:
        m = opt_result.metrics
        optimizer_metrics = {
            "accepted": m.accepted,
            "rejected": m.rejected,
            "merge_attempts": m.merge_attempts,
            "cache_hits": m.cache_hits,
            "cache_misses": m.cache_misses,
            "acceptance_rate": m.acceptance_rate,
            "cache_hit_rate": m.cache_hit_rate,
            "best_score_history": list(m.best_score_history),
        } if m is not None else None

        candidates_dump = [
            {
                "index": c.index,
                "origin": c.origin,
                "parent_index": c.parent_index,
                "merge_parent_indices": list(c.merge_parent_indices)
                    if c.merge_parent_indices else None,
                "on_pareto_front": c.on_pareto_front,
                "mean_score": c.mean_score,
                "min_score": c.min_score,
                "n_val_scores": len(c.scores),
                "n_train_scores": len(c.train_scores),
                "val_scores": dict(c.scores),
                "train_scores": dict(c.train_scores),
                "agents": [
                    {"name": a.name, "instruction": a.instruction}
                    for a in c.config.agents
                ],
            }
            for c in opt_result.all_candidates
        ]
    else:
        optimizer_metrics = None
        candidates_dump = []

    metrics: dict[str, float] = {
        "baseline_accuracy": baseline_acc,
        "optimized_accuracy": optimized_acc,
        "improvement": optimized_acc - baseline_acc,
    }
    if train_eval_tasks:
        metrics["train_accuracy"] = sum(t.correct for t in train_eval_tasks) / len(train_eval_tasks)
    if opt_result is not None:
        metrics["val_accuracy"] = opt_result.best_score

    result = BenchmarkResult(
        benchmark="aime_math",
        metrics=metrics,
        iterations=opt_result.iterations if opt_result else 0,
        total_evaluation_runs=opt_result.total_evaluation_runs if opt_result else None,
        cost=CostSummary(
            prompt_tokens=opt_result.total_prompt_tokens if opt_result else 0,
            completion_tokens=opt_result.total_completion_tokens if opt_result else 0,
            total_tokens=(
                (opt_result.total_prompt_tokens + opt_result.total_completion_tokens)
                if opt_result
                else 0
            ),
        ),
        per_task=optimized_tasks,
        seed_config=seed_config.model_dump(),
        optimized_config=(
            opt_result.best_config.model_dump() if opt_result else None
        ),
        optimizer_metrics=optimizer_metrics,
        candidates=candidates_dump,
    )

    path = save_result(result, Path(settings.output_dir))
    report(result)
    _log.info("Results saved to {}", path)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AIME Math benchmark")
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--max-evaluations", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--test-repeats", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--eval-concurrency", type=int, default=None)
    parser.add_argument("--max-output-tokens", type=int, default=None)
    parser.add_argument("--skip-baseline-test", action="store_true", default=None)
    parser.add_argument("--baseline-accuracy", type=float, default=None)
    parser.add_argument("--eval-best-on-train", action="store_true", default=None)
    parser.add_argument("--checkpoint-path", default=None)
    args = parser.parse_args()

    raw = vars(args)
    overrides = {k: v for k, v in raw.items() if v is not None}
    overrides = {k.replace("-", "_"): v for k, v in overrides.items()}
    settings = AimeMathSettings(**overrides)

    asyncio.run(main(settings))
