"""Example: custom callbacks and MetricsCallback.

Demonstrates how to implement a callback for logging and how to use the
built-in MetricsCallback to collect optimization statistics.
"""

import asyncio

from fedotmas import MAW, Optimizer
from fedotmas.maw.models import MAWAgentConfig, MAWConfig, MAWStepConfig
from fedotmas.optimize import (
    MetricsCallback,
    OptimizationCallback,
    OptimizationConfig,
)
from fedotmas.optimize._result import OptimizationResult
from fedotmas.optimize._state import Candidate, OptimizationState

SEED_CONFIG = MAWConfig(
    agents=[
        MAWAgentConfig(
            name="planner",
            instruction="Create a structured plan for the given task.",
            output_key="plan",
        ),
        MAWAgentConfig(
            name="executor",
            instruction="Execute the plan and produce a detailed result.",
            output_key="result",
        ),
    ],
    pipeline=MAWStepConfig(
        type="sequential",
        children=[
            MAWStepConfig(type="agent", agent_name="planner"),
            MAWStepConfig(type="agent", agent_name="executor"),
        ],
    ),
)

TRAINSET = [
    "Design a REST API for a todo application",
    "Plan a database migration strategy for a legacy system",
    "Outline a CI/CD pipeline for a Python monorepo",
]


class LoggingCallback(OptimizationCallback):
    """Prints structured events to stdout."""

    def __init__(self) -> None:
        self.log: list[str] = []

    def on_iteration_start(self, iteration: int, state: OptimizationState) -> None:
        msg = f"[iter {iteration}] start | candidates={len(state.candidates)}"
        self.log.append(msg)
        print(msg)

    def on_candidate_accepted(self, child: Candidate, parent: Candidate) -> None:
        msg = (
            f"  ACCEPTED #{child.index} "
            f"(score={child.mean_score or 0:.3f}) "
            f"from parent #{parent.index}"
        )
        self.log.append(msg)
        print(msg)

    def on_candidate_rejected(self, child: Candidate, parent: Candidate) -> None:
        msg = (
            f"  rejected #{child.index} "
            f"(score={child.mean_score or 0:.3f}) "
            f"vs parent #{parent.index}"
        )
        self.log.append(msg)
        print(msg)

    def on_optimization_end(self, result: OptimizationResult) -> None:
        msg = (
            f"[done] best_score={result.best_score:.3f} iterations={result.iterations}"
        )
        self.log.append(msg)
        print(msg)


async def main() -> None:
    maw = MAW()

    logging_cb = LoggingCallback()
    metrics_cb = MetricsCallback()

    opt = Optimizer(
        maw,
        criteria="API design quality, completeness, and best practices",
        config=OptimizationConfig(
            seed=42,
            minibatch_size=2,
            use_merge=False,
            max_iterations=5,
            patience=3,
        ),
        callbacks=[logging_cb, metrics_cb],
    )

    await opt.optimize(TRAINSET, seed_config=SEED_CONFIG)

    print("\n--- Metrics Summary ---")
    m = metrics_cb.metrics
    print(f"Accepted: {m.accepted}")
    print(f"Rejected: {m.rejected}")
    print(f"Acceptance rate: {m.acceptance_rate:.1%}")
    print(f"Cache hits: {m.cache_hits}  misses: {m.cache_misses}")
    print(f"Cache hit rate: {m.cache_hit_rate:.1%}")
    print(f"Best score history: {[f'{s:.3f}' for s in m.best_score_history]}")
    print(f"Log entries: {len(logging_cb.log)}")


if __name__ == "__main__":
    asyncio.run(main())
