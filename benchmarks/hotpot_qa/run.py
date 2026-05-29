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
from fedotmas.plugins import LLMRoutingPlugin

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _utils import BenchmarkResult, CostSummary, TaskResult, save_result
from dataset import load_hotpot_dataset
from scorer import HotpotQAScorer, exact_match
from settings import HotpotQASettings

_log = get_logger("fmbench.hotpot_qa")

_SEED_CONFIG_PATH = Path(__file__).parent / "seed_config.json"


def _load_seed_config(settings: HotpotQASettings) -> MAWConfig:
    if not _SEED_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Seed config not found at {_SEED_CONFIG_PATH}. "
            f"Run `python benchmarks/hotpot_qa/generate_seed.py` first."
        )
    _log.info("Loading seed config from {}", _SEED_CONFIG_PATH)
    config = MAWConfig.model_validate_json(_SEED_CONFIG_PATH.read_text())
    # settings.solver_model takes precedence over the model embedded in the
    # seed config: the user sets it explicitly (env var / CLI), while the seed
    # config's model is a generation-time artefact that's easy to overlook.
    original_models = sorted({a.model for a in config.agents if a.model})
    for agent in config.agents:
        agent.model = settings.solver_model
    if original_models and original_models != [settings.solver_model]:
        _log.info(
            "Overriding seed_config agent models {} with settings.solver_model={!r}. "
            "Note: instructions in the seed may have been tuned for the original model.",
            original_models,
            settings.solver_model,
        )
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
    agent = next((a for a in config.agents if a.name == name), None)
    if agent is None:
        raise ValueError(
            f"Pipeline references agent {name!r} but it is not defined in "
            f"config.agents (available: {[a.name for a in config.agents]})"
        )
    return agent.output_key


async def _solve_one(
    i: int,
    task: Task,
    config: MAWConfig,
    answer_key: str,
    maw: MAW,
    scorer: HotpotQAScorer,
    sem: asyncio.Semaphore,
    stage: str,
    total: int,
    progress: dict[str, float],
    routing_plugin: LLMRoutingPlugin | None = None,
) -> TaskResult:
    async with sem:
        _log.info("[{}] Task {}/{} — solving...", stage, i + 1, total)
        plugins = [routing_plugin] if routing_plugin is not None else None
        run = None
        try:
            run = await Controller(maw).run(task.input, config=config, plugins=plugins)
            if run.status == "error":
                err_msg = run.error.message if run.error else "unknown error"
                _log.warning(
                    "[{}] Task {}/{} pipeline error: {}", stage, i + 1, total, err_msg
                )
                scoring = None
                output = f"ERROR: {err_msg}"
            else:
                scoring = await scorer.evaluate(task, run.state)
                output = str(run.state.get(answer_key, ""))
        except Exception as exc:
            _log.warning("[{}] Task {}/{} failed: {}", stage, i + 1, total, exc)
            scoring = None
            output = f"ERROR: {exc}"

        score = scoring.score if scoring else 0.0
        if routing_plugin is not None and run is not None and run.invocation_id is not None:
            # Failed pipelines get score=0.0 (set above), and any step-level
            # records appended before the failure still get backfilled — the
            # router needs to learn that those step choices led to a
            # zero-score outcome.
            routing_plugin.commit_task_score(run.invocation_id, score)
        # EM and F1 are independent: bag-of-words F1 can hit 1.0 on
        # word-order swaps that fail normalized string equality, so compute
        # EM via the same normalization the scorer uses rather than score==1.0.
        is_em = bool(scoring) and exact_match(output, task.expected or "")
        progress["done"] += 1
        progress["em"] += float(is_em)
        progress["f1_sum"] += score
        _log.info(
            "[{}] Task {}/{} {} | f1={:.2f} expected={!r} | "
            "running em={:.0f}/{:.0f} ({:.1%}) f1={:.2f}",
            stage, i + 1, total,
            "✓" if is_em else "✗",
            score, task.expected,
            progress["em"], progress["done"],
            progress["em"] / progress["done"],
            progress["f1_sum"] / progress["done"],
        )

        return TaskResult(
            task_id=str(i),
            input=task.input[:200],
            expected=task.expected,
            output=output,
            score=score,
            correct=is_em,
        )


async def evaluate_on(
    config: MAWConfig,
    tasks: list[Task],
    maw: MAW,
    scorer: HotpotQAScorer,
    stage: str = "eval",
    concurrency: int = 1,
    routing_plugin: LLMRoutingPlugin | None = None,
) -> list[TaskResult]:
    total = len(tasks)
    sem = asyncio.Semaphore(max(1, concurrency))
    progress = {"done": 0.0, "em": 0.0, "f1_sum": 0.0}
    answer_key = _final_output_key(config)
    _log.info(
        "[{}] Evaluating {} tasks with concurrency={}", stage, total, concurrency
    )
    coros = [
        _solve_one(
            i, t, config, answer_key, maw, scorer, sem, stage, total, progress,
            routing_plugin=routing_plugin,
        )
        for i, t in enumerate(tasks)
    ]
    return await asyncio.gather(*coros)


def report(result: BenchmarkResult) -> None:
    m = result.metrics
    _log.info("HotpotQA Benchmark Results")
    _log.info("Iterations:          {}", result.iterations)
    _log.info("Baseline EM:         {:.1%}", m.get("baseline_accuracy", 0))
    _log.info("Optimized EM:        {:.1%}", m.get("optimized_accuracy", 0))
    _log.info("Improvement (EM):    {:+.1%}", m.get("improvement", 0))
    _log.info("Baseline F1:         {:.3f}", m.get("baseline_f1", 0))
    _log.info("Optimized F1:        {:.3f}", m.get("optimized_f1", 0))
    if "train_accuracy" in m:
        _log.info("Train EM:            {:.1%}", m["train_accuracy"])
        _log.info("Val F1:              {:.3f} (best on full valset)", m.get("val_f1", 0))
    if result.cost:
        _log.info("Total tokens:        {:,}", result.cost.total_tokens)


def _f1_mean(tasks: list[TaskResult]) -> float:
    return sum(t.score for t in tasks) / len(tasks) if tasks else 0.0


def _em_rate(tasks: list[TaskResult]) -> float:
    return sum(t.correct for t in tasks) / len(tasks) if tasks else 0.0


async def main(
    settings: HotpotQASettings,
    routing_plugin: LLMRoutingPlugin | None = None,
) -> BenchmarkResult:
    trainset, valset, testset = load_hotpot_dataset(
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
    # Scorer is created once from seed_config's output_key and reused across all
    # candidates. Safe while InstructionMutator only edits instructions; if a
    # future mutator changes pipeline topology / agent names / output_keys, the
    # scorer will read a stale key — make Scorer.evaluate accept the current
    # config (or recompute the key) before adding such mutators.
    scorer = HotpotQAScorer(output_key=answer_key)

    train_eval_tasks: list[TaskResult] = []

    if settings.max_iterations == 0:
        _log.info("max_iterations=0 — skipping optimizer, evaluating seed on testset only")
        opt_result = None
        baseline_tasks = await evaluate_on(
            seed_config, testset, maw, scorer,
            stage="baseline", concurrency=settings.concurrency,
            routing_plugin=routing_plugin,
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
                routing_plugin=routing_plugin,
            )

        if settings.eval_best_on_train:
            _log.info("Evaluating best on full trainset ({} tasks)", len(trainset))
            train_eval_tasks = await evaluate_on(
                opt_result.best_config, trainset, maw, scorer,
                stage="train-final", concurrency=settings.concurrency,
                routing_plugin=routing_plugin,
            )

    if baseline_tasks:
        baseline_em = _em_rate(baseline_tasks)
        baseline_f1 = _f1_mean(baseline_tasks)
    elif settings.baseline_accuracy is not None:
        baseline_em = settings.baseline_accuracy
        baseline_f1 = 0.0
    else:
        baseline_em = 0.0
        baseline_f1 = 0.0

    optimized_em = _em_rate(optimized_tasks)
    optimized_f1 = _f1_mean(optimized_tasks)

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
        "baseline_accuracy": baseline_em,
        "optimized_accuracy": optimized_em,
        "improvement": optimized_em - baseline_em,
        "baseline_f1": baseline_f1,
        "optimized_f1": optimized_f1,
    }
    if train_eval_tasks:
        metrics["train_accuracy"] = _em_rate(train_eval_tasks)
        metrics["train_f1"] = _f1_mean(train_eval_tasks)
    if opt_result is not None:
        metrics["val_f1"] = opt_result.best_score

    result = BenchmarkResult(
        benchmark="hotpot_qa",
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
    parser = argparse.ArgumentParser(description="HotpotQA benchmark")
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

    overrides = {k: v for k, v in vars(args).items() if v is not None}
    settings = HotpotQASettings(**overrides)

    asyncio.run(main(settings))
