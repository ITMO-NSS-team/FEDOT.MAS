from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from pathlib import Path

from fedotmas.common.logging import get_logger
from fedotmas.control._controller import Controller
from fedotmas.control._run import ControlledRun
from fedotmas.maw.maw import MAW
from fedotmas.maw.models import MAWConfig
from fedotmas.optimize._callbacks import (
    CallbackDispatcher,
    MetricsCallback,
    OptimizationCallback,
)
from fedotmas.optimize._config import OptimizationConfig
from fedotmas.optimize._mutators._protocol import Mutator
from fedotmas.optimize._result import OptimizationResult
from fedotmas.optimize._scoring import Scorer, ScoringResult
from fedotmas.optimize._state import (
    Candidate,
    OptimizationState,
    TaskResult,
    config_hash,
    find_common_ancestor,
    is_ancestor_of,
)
from fedotmas.optimize._stopping import SignalStopper, Stopper
from fedotmas.optimize._strategies import (
    BatchSampler,
    CandidateSelector,
    ComponentSelector,
)

_log = get_logger("fedotmas.optimize._engine")


@dataclass
class _MergeSchedule:
    """Tracks merge scheduling state (GEPA merge scheduling logic)."""

    merges_due: int = 0
    total_tested: int = 0
    last_found_new: bool = False


@dataclass
class _LoopContext:
    maw: MAW
    scorer: Scorer
    mutator: Mutator
    candidate_selector: CandidateSelector
    batch_sampler: BatchSampler
    component_selector: ComponentSelector
    dispatcher: CallbackDispatcher
    metrics_cb: MetricsCallback
    state: OptimizationState
    cfg: OptimizationConfig
    rng: random.Random
    trainset: list[str]
    valset: list[str]
    seed: Candidate | None
    merge: _MergeSchedule = field(default_factory=_MergeSchedule)
    merged_pairs: set[tuple[int, int]] = field(default_factory=set)


async def run_optimization(
    *,
    maw: MAW,
    seed_config: MAWConfig,
    trainset: list[str],
    valset: list[str],
    scorer: Scorer,
    mutator: Mutator,
    candidate_selector: CandidateSelector,
    batch_sampler: BatchSampler,
    component_selector: ComponentSelector,
    stopper: Stopper,
    config: OptimizationConfig | None = None,
    callbacks: list[OptimizationCallback] | None = None,
) -> OptimizationResult:
    cfg = config or OptimizationConfig()
    rng = cfg.rng

    dispatcher = CallbackDispatcher()
    metrics_cb = MetricsCallback()
    dispatcher.add(metrics_cb)
    for cb in callbacks or []:
        dispatcher.add(cb)

    signal_stopper: SignalStopper | None = None
    if cfg.graceful_shutdown:
        signal_stopper = SignalStopper()

    state, seed, iteration = _setup_state(cfg, seed_config)

    total_eval_runs = 0

    if seed is not None and not seed.scores:
        _log.info("Evaluating seed candidate on valset ({} tasks)", len(valset))
        eval_runs = await _evaluate_candidate(
            maw, scorer, seed, valset, state, cfg, metrics_cb
        )
        total_eval_runs += eval_runs
        state.update_pareto_front()
        dispatcher.on_candidate_evaluated(seed, valset)

        _log.info(
            "Seed | mean_score={:.3f} min_score={:.3f}",
            seed.mean_score or 0.0,
            seed.min_score or 0.0,
        )

    consecutive_failures = 0

    def _should_stop() -> bool:
        if signal_stopper is not None and signal_stopper.should_stop(state, iteration):
            return True
        return stopper.should_stop(state, iteration)

    reset_fn = getattr(stopper, "reset", None)
    if reset_fn is not None:
        reset_fn()

    ctx = _LoopContext(
        maw=maw,
        scorer=scorer,
        mutator=mutator,
        candidate_selector=candidate_selector,
        batch_sampler=batch_sampler,
        component_selector=component_selector,
        dispatcher=dispatcher,
        metrics_cb=metrics_cb,
        state=state,
        cfg=cfg,
        rng=rng,
        trainset=trainset,
        valset=valset,
        seed=seed,
    )

    try:
        if signal_stopper is not None:
            signal_stopper.install()

        while not _should_stop():
            iteration += 1
            state.iteration = iteration
            _log.info("=== Iteration {} ===", iteration)
            dispatcher.on_iteration_start(iteration, state)

            eval_runs, consecutive_failures = await _run_loop_step(
                ctx, iteration, consecutive_failures
            )
            total_eval_runs += eval_runs

            dispatcher.on_iteration_end(iteration, state)
            _checkpoint(state, cfg)

    finally:
        if signal_stopper is not None:
            signal_stopper.uninstall()

    return _build_result(
        state, seed, iteration, total_eval_runs, mutator, metrics_cb, dispatcher
    )


def _checkpoint(state: OptimizationState, cfg: OptimizationConfig) -> None:
    cp = Path(cfg.checkpoint_path) if cfg.checkpoint_path else None
    if cp is not None:
        state.save(cp)


async def _run_loop_step(
    ctx: _LoopContext,
    iteration: int,
    consecutive_failures: int,
) -> tuple[int, int]:
    """Run one loop step: merge OR mutation. Returns (eval_runs, consecutive_failures)."""
    cfg = ctx.cfg
    ms = ctx.merge

    # 1) Merge first if scheduled (replaces mutation this iteration)
    if (
        cfg.use_merge
        and ms.merges_due > 0
        and ms.last_found_new
        and len(ctx.state.get_pareto_candidates()) >= 2
    ):
        merge_eval, merge_accepted = await _try_merge(ctx)
        ms.last_found_new = False
        if merge_accepted:
            ms.merges_due -= 1
            ms.total_tested += 1
        return merge_eval, consecutive_failures

    # 2) Mutation
    eval_runs, accepted, fail_count = await _run_iteration(
        ctx, iteration, consecutive_failures
    )
    ms.last_found_new = accepted

    # 3) Schedule merge after successful mutation
    if accepted and cfg.use_merge and ms.total_tested < cfg.max_merge_attempts:
        ms.merges_due += 1

    return eval_runs, fail_count


def _setup_state(
    cfg: OptimizationConfig, seed_config: MAWConfig
) -> tuple[OptimizationState, Candidate | None, int]:
    cp = Path(cfg.checkpoint_path) if cfg.checkpoint_path else None
    if cp is not None and cp.exists():
        _log.info("Resuming from checkpoint: {}", cp)
        state = OptimizationState.load(cp)
        seed = state.candidates[0] if state.candidates else None
        iteration = state.iteration
        return state, seed, iteration

    state = OptimizationState()
    seed = state.add_candidate(seed_config, origin="seed")
    return state, seed, 0


async def _run_iteration(
    ctx: _LoopContext,
    iteration: int,
    consecutive_failures: int,
) -> tuple[int, bool, int]:
    """Run one iteration. Returns (eval_runs, accepted, consecutive_failures)."""
    eval_runs = 0
    state = ctx.state
    cfg = ctx.cfg

    evaluated = [c for c in state.candidates if c.scores]
    if not evaluated:
        if ctx.seed is not None:
            evaluated = [ctx.seed]
        else:
            return 0, False, consecutive_failures

    parent = ctx.candidate_selector.select(evaluated)
    batch = ctx.batch_sampler.sample(ctx.trainset, cfg.minibatch_size)
    components = ctx.component_selector.select(parent.config, iteration - 1)

    _log.info(
        "Parent #{} (score={:.3f}) | components={} | batch={}",
        parent.index,
        parent.mean_score or 0.0,
        components,
        batch,
    )

    runs = await _evaluate_candidate(
        ctx.maw, ctx.scorer, parent, batch, state, cfg, ctx.metrics_cb
    )
    eval_runs += runs

    try:
        new_config = await ctx.mutator.mutate(parent, components, batch)
    except Exception as e:
        consecutive_failures += 1
        _log.warning(
            "Mutator failed ({}/{}): {}: {}",
            consecutive_failures,
            cfg.max_consecutive_failures,
            type(e).__name__,
            e,
        )
        if consecutive_failures >= cfg.max_consecutive_failures:
            _log.warning(
                "Max consecutive failures reached — shuffling agent order as fallback. "
                "Last error: {}: {}. Consider increasing llm_timeout or checking LLM connectivity.",
                type(e).__name__,
                e,
            )
            agents = list(parent.config.agents)
            ctx.rng.shuffle(agents)
            new_config = parent.config.model_copy(update={"agents": agents})
            consecutive_failures = 0
        else:
            return eval_runs, False, consecutive_failures

    new_hash = config_hash(new_config)
    if new_hash == parent.config_hash:
        _log.info("Mutation produced identical config, skipping")
        return eval_runs, False, consecutive_failures

    child = state.add_candidate(
        new_config, parent_index=parent.index, origin="mutation"
    )
    runs = await _evaluate_candidate(
        ctx.maw, ctx.scorer, child, batch, state, cfg, ctx.metrics_cb
    )
    eval_runs += runs
    ctx.dispatcher.on_candidate_evaluated(child, batch)

    batch_tasks = set(batch)
    parent_batch_score = _mean_score_on(parent, batch_tasks)
    child_batch_score = _mean_score_on(child, batch_tasks)

    if child_batch_score > parent_batch_score:
        _log.info(
            "Accepted | child #{} score={:.3f} > parent score={:.3f}",
            child.index,
            child_batch_score,
            parent_batch_score,
        )
        consecutive_failures = 0
        ctx.dispatcher.on_candidate_accepted(child, parent)

        runs = await _evaluate_candidate(
            ctx.maw, ctx.scorer, child, ctx.valset, state, cfg, ctx.metrics_cb
        )
        eval_runs += runs
        state.update_pareto_front()
        return eval_runs, True, consecutive_failures
    else:
        _log.info(
            "Rejected | child #{} score={:.3f} <= parent score={:.3f}",
            child.index,
            child_batch_score,
            parent_batch_score,
        )
        ctx.dispatcher.on_candidate_rejected(child, parent)
        return eval_runs, False, consecutive_failures


async def _try_merge(ctx: _LoopContext) -> tuple[int, bool]:
    """Attempt a merge. Returns (eval_runs, accepted)."""
    state = ctx.state
    cfg = ctx.cfg
    eval_runs = 0

    pareto = state.get_pareto_candidates()
    if len(pareto) < 2:
        return 0, False

    # Select a valid pair with filtering (like GEPA merge.py:69-116)
    pair: list[Candidate] | None = None
    for _ in range(10):
        sample = ctx.rng.sample(pareto, 2)
        key = (
            min(sample[0].index, sample[1].index),
            max(sample[0].index, sample[1].index),
        )
        # Triplet dedup: don't re-merge the same pair
        if key in ctx.merged_pairs:
            continue
        # Don't merge direct ancestors
        if is_ancestor_of(sample[0], sample[1], state.candidates):
            continue
        if is_ancestor_of(sample[1], sample[0], state.candidates):
            continue
        pair = sample
        ctx.merged_pairs.add(key)
        break

    if pair is None:
        _log.info("No valid merge pair found after filtering")
        return 0, False

    ctx.dispatcher.on_merge_attempted((pair[0], pair[1]))

    ancestor = find_common_ancestor(pair[0], pair[1], state.candidates)
    if ancestor is not None:
        # Ancestor must not be better than children
        anc_score = ancestor.mean_score or 0.0
        if anc_score > (pair[0].mean_score or 0.0) or anc_score > (
            pair[1].mean_score or 0.0
        ):
            _log.info(
                "Skipping merge: ancestor #{} score {:.3f} > child",
                ancestor.index,
                anc_score,
            )
            return 0, False
        _log.info(
            "Genealogy merge: ancestor #{} for pair ({}, {})",
            ancestor.index,
            pair[0].index,
            pair[1].index,
        )
        merged_config = await ctx.mutator.genealogy_merge(
            ancestor, pair[0], pair[1], ctx.trainset
        )
    else:
        merged_config = await ctx.mutator.merge(pair[0], pair[1], ctx.trainset)

    merged_hash = config_hash(merged_config)
    if merged_hash == pair[0].config_hash or merged_hash == pair[1].config_hash:
        _log.info("Merge produced config identical to a parent, skipping")
        return 0, False

    merged = state.add_candidate(
        merged_config,
        parent_index=pair[0].index,
        merge_parent_indices=(pair[0].index, pair[1].index),
        origin="merge",
    )
    runs = await _evaluate_candidate(
        ctx.maw, ctx.scorer, merged, ctx.valset, state, cfg, ctx.metrics_cb
    )
    eval_runs += runs
    ctx.dispatcher.on_candidate_evaluated(merged, ctx.valset)

    # Acceptance check: merged must be >= max(parents) (GEPA engine.py:479)
    parent_best = max(pair[0].mean_score or 0.0, pair[1].mean_score or 0.0)
    merged_score = merged.mean_score or 0.0

    if merged_score >= parent_best:
        _log.info(
            "Merge accepted #{} | score={:.3f} >= max parent {:.3f}",
            merged.index,
            merged_score,
            parent_best,
        )
        state.update_pareto_front()
        return eval_runs, True
    else:
        _log.info(
            "Merge rejected #{} | score={:.3f} < max parent {:.3f}",
            merged.index,
            merged_score,
            parent_best,
        )
        ctx.dispatcher.on_candidate_rejected(merged, pair[0])
        return eval_runs, False


def _build_result(
    state: OptimizationState,
    seed: Candidate | None,
    iteration: int,
    total_eval_runs: int,
    mutator: Mutator,
    metrics_cb: MetricsCallback,
    dispatcher: CallbackDispatcher,
) -> OptimizationResult:
    best = state.best_candidate()
    if best is None and seed is not None:
        best = seed
    if best is None:
        raise RuntimeError("No candidates were evaluated")

    _log.info(
        "Optimization complete | iterations={} candidates={} best_score={:.3f}",
        iteration,
        len(state.candidates),
        best.mean_score or 0.0,
    )

    result = OptimizationResult(
        best_config=best.config,
        best_score=best.mean_score or 0.0,
        all_candidates=state.candidates,
        iterations=iteration,
        total_evaluation_runs=total_eval_runs,
        total_prompt_tokens=mutator.token_usage[0],
        total_completion_tokens=mutator.token_usage[1],
        metrics=metrics_cb.metrics,
    )
    dispatcher.on_optimization_end(result)
    return result


async def _evaluate_candidate(
    maw: MAW,
    scorer: Scorer,
    candidate: Candidate,
    tasks: list[str],
    state: OptimizationState,
    config: OptimizationConfig,
    metrics_cb: MetricsCallback | None = None,
) -> int:
    """Returns number of new evaluation runs performed."""
    tasks_to_run: list[str] = []
    for task in tasks:
        cached = state.cache.get(candidate.config_hash, task)
        if cached is not None:
            state.record_task_result(candidate, cached)
            if metrics_cb is not None:
                metrics_cb.metrics.cache_hits += 1
        else:
            tasks_to_run.append(task)
            if metrics_cb is not None:
                metrics_cb.metrics.cache_misses += 1

    if not tasks_to_run:
        return 0

    runs: list[ControlledRun | BaseException] = await asyncio.gather(
        *[Controller(maw).run(task, config=candidate.config) for task in tasks_to_run],
        return_exceptions=True,
    )

    max_failures = config.max_consecutive_failures
    consecutive_errors = 0

    for task, run in zip(tasks_to_run, runs):
        if isinstance(run, BaseException):
            consecutive_errors += 1
            result = TaskResult(
                task=task,
                state={},
                score=0.0,
                feedback=f"Pipeline failed: {run}",
                error=True,
            )
            state.record_task_result(candidate, result)
            if consecutive_errors >= max_failures:
                _log.warning(
                    "Skipping remaining tasks after {} consecutive failures",
                    max_failures,
                )
                break
            continue
        else:
            consecutive_errors = 0
            try:
                scoring: ScoringResult = await scorer.evaluate(task, run.state)
                result = TaskResult(
                    task=task,
                    state=run.state,
                    score=scoring.score,
                    feedback=scoring.feedback,
                )
            except Exception as e:
                _log.warning("Scoring failed for task '{}': {}", task, e)
                result = TaskResult(
                    task=task,
                    state=run.state,
                    score=0.0,
                    feedback=f"Scoring failed: {e}",
                    error=True,
                )
        state.record_task_result(candidate, result)

    return len(tasks_to_run)


def _mean_score_on(candidate: Candidate, tasks: set[str]) -> float:
    scores = [candidate.scores[t] for t in tasks if t in candidate.scores]
    if not scores:
        return 0.0
    return sum(scores) / len(scores)
