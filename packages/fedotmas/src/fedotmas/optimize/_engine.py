from __future__ import annotations

import asyncio
import random

from fedotmas.common.logging import get_logger
from fedotmas.control._controller import Controller
from fedotmas.control._run import ControlledRun
from fedotmas.maw.maw import MAW
from fedotmas.maw.models import MAWConfig
from fedotmas.optimize._proposer import Proposer
from fedotmas.optimize._result import OptimizationResult
from fedotmas.optimize._scoring import Scorer, ScoringResult
from fedotmas.optimize._state import (
    Candidate,
    OptimizationState,
    TaskResult,
    config_hash,
)
from fedotmas.optimize._stopping import Stopper
from fedotmas.optimize._strategies import (
    BatchSampler,
    CandidateSelector,
    ComponentSelector,
)

_log = get_logger("fedotmas.optimize._engine")


async def run_optimization(
    *,
    maw: MAW,
    seed_config: MAWConfig,
    trainset: list[str],
    valset: list[str],
    scorer: Scorer,
    proposer: Proposer,
    candidate_selector: CandidateSelector,
    batch_sampler: BatchSampler,
    component_selector: ComponentSelector,
    stopper: Stopper,
    use_merge: bool,
    max_merge_attempts: int,
    minibatch_size: int,
) -> OptimizationResult:
    reset_fn = getattr(stopper, "reset", None)
    if reset_fn is not None:
        reset_fn()

    state = OptimizationState()
    total_eval_runs = 0

    seed = state.add_candidate(seed_config, origin="seed")
    _log.info("Evaluating seed candidate on full trainset ({} tasks)", len(trainset))
    eval_runs = await _evaluate_candidate(maw, scorer, seed, trainset, state)
    total_eval_runs += eval_runs
    state.update_pareto_front()

    _log.info(
        "Seed | mean_score={:.3f} min_score={:.3f}",
        seed.mean_score or 0.0,
        seed.min_score or 0.0,
    )

    iteration = 0
    merge_attempts = 0
    last_accepted = False

    while not stopper.should_stop(state, iteration):
        iteration += 1
        _log.info("=== Iteration {} ===", iteration)

        evaluated = [c for c in state.candidates if c.scores]
        if not evaluated:
            evaluated = [seed]
        parent = candidate_selector.select(evaluated)

        batch = batch_sampler.sample(trainset, minibatch_size)
        components = component_selector.select(parent.config, iteration - 1)

        _log.info(
            "Parent #{} (score={:.3f}) | components={} | batch={}",
            parent.index,
            parent.mean_score or 0.0,
            components,
            batch,
        )

        eval_runs = await _evaluate_candidate(maw, scorer, parent, batch, state)
        total_eval_runs += eval_runs

        new_config = await proposer.propose_mutation(parent, components, batch)
        new_hash = config_hash(new_config)

        if new_hash == parent.config_hash:
            _log.info("Mutation produced identical config, skipping")
            last_accepted = False
            continue

        child = state.add_candidate(
            new_config, parent_index=parent.index, origin="mutation"
        )
        eval_runs = await _evaluate_candidate(maw, scorer, child, batch, state)
        total_eval_runs += eval_runs

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
            last_accepted = True

            eval_runs = await _evaluate_candidate(maw, scorer, child, valset, state)
            total_eval_runs += eval_runs
            state.update_pareto_front()
        else:
            _log.info(
                "Rejected | child #{} score={:.3f} <= parent score={:.3f}",
                child.index,
                child_batch_score,
                parent_batch_score,
            )
            last_accepted = False

        if (
            use_merge
            and last_accepted
            and merge_attempts < max_merge_attempts
            and len(state.get_pareto_candidates()) >= 2
        ):
            merge_attempts += 1
            _log.info(
                "Attempting merge (attempt {}/{})", merge_attempts, max_merge_attempts
            )

            pareto = state.get_pareto_candidates()
            pair = random.sample(pareto, 2)
            merged_config = await proposer.propose_merge(pair[0], pair[1], trainset)
            merged_hash = config_hash(merged_config)

            if (
                merged_hash != pair[0].config_hash
                and merged_hash != pair[1].config_hash
            ):
                merged = state.add_candidate(
                    merged_config,
                    parent_index=pair[0].index,
                    origin="merge",
                )
                eval_runs = await _evaluate_candidate(
                    maw, scorer, merged, valset, state
                )
                total_eval_runs += eval_runs
                state.update_pareto_front()

                _log.info(
                    "Merge #{} | score={:.3f}",
                    merged.index,
                    merged.mean_score or 0.0,
                )

    best = state.best_candidate()
    if best is None:
        best = seed

    _log.info(
        "Optimization complete | iterations={} candidates={} best_score={:.3f}",
        iteration,
        len(state.candidates),
        best.mean_score or 0.0,
    )

    return OptimizationResult(
        best_config=best.config,
        best_score=best.mean_score or 0.0,
        all_candidates=state.candidates,
        iterations=iteration,
        total_evaluation_runs=total_eval_runs,
        total_prompt_tokens=proposer.total_prompt_tokens,
        total_completion_tokens=proposer.total_completion_tokens,
    )


async def _evaluate_candidate(
    maw: MAW,
    scorer: Scorer,
    candidate: Candidate,
    tasks: list[str],
    state: OptimizationState,
) -> int:
    """Returns number of new evaluation runs performed."""
    tasks_to_run: list[str] = []
    for task in tasks:
        cached = state.cache.get(candidate.config_hash, task)
        if cached is not None:
            state.record_task_result(candidate, cached)
        else:
            tasks_to_run.append(task)

    if not tasks_to_run:
        return 0

    runs: list[ControlledRun | BaseException] = await asyncio.gather(
        *[Controller(maw).run(task, config=candidate.config) for task in tasks_to_run],
        return_exceptions=True,
    )

    for task, run in zip(tasks_to_run, runs):
        if isinstance(run, BaseException):
            result = TaskResult(
                task=task,
                state={},
                score=0.0,
                feedback=f"Pipeline failed: {run}",
                error=True,
            )
        else:
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
