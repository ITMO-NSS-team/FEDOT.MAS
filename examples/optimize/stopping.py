"""Example: stopping criteria.

Demonstrates the various stopping strategies: patience (NoImprovement),
ScoreThreshold, MaxEvaluations, and SignalStopper.
"""

import asyncio

from fedotmas import MAW, Optimizer
from fedotmas.maw.models import MAWAgentConfig, MAWConfig, MAWStepConfig
from fedotmas.optimize import OptimizationConfig

SEED_CONFIG = MAWConfig(
    agents=[
        MAWAgentConfig(
            name="drafter",
            model="openai/gpt-4o-mini",
            instruction="Draft a response to the given task.",
            output_key="draft",
        ),
        MAWAgentConfig(
            name="refiner",
            model="openai/gpt-4o-mini",
            instruction="Refine the draft for clarity and completeness.",
            output_key="final",
        ),
    ],
    pipeline=MAWStepConfig(
        type="sequential",
        children=[
            MAWStepConfig(type="agent", agent_name="drafter"),
            MAWStepConfig(type="agent", agent_name="refiner"),
        ],
    ),
)

TRAINSET = [
    "Explain the CAP theorem in distributed systems",
    "Describe how garbage collection works in JVM",
    "Summarize the Raft consensus algorithm",
]


async def run_with_patience() -> None:
    """Stop after N iterations without improvement."""
    print("--- patience=2 (NoImprovement) ---")
    maw = MAW()
    opt = Optimizer(
        maw,
        criteria="Technical accuracy and clarity",
        config=OptimizationConfig(
            seed=42,
            minibatch_size=2,
            use_merge=False,
            max_iterations=20,
            patience=2,
        ),
    )
    result = await opt.optimize(TRAINSET, seed_config=SEED_CONFIG)
    print(
        f"  Stopped at iteration {result.iterations}, best_score={result.best_score:.3f}"
    )


async def run_with_score_threshold() -> None:
    """Stop when score reaches a threshold."""
    print("\n--- score_threshold=0.85 ---")
    maw = MAW()
    opt = Optimizer(
        maw,
        criteria="Technical accuracy and clarity",
        config=OptimizationConfig(
            seed=42,
            minibatch_size=2,
            use_merge=False,
            max_iterations=20,
            patience=20,
            score_threshold=0.85,
        ),
    )
    result = await opt.optimize(TRAINSET, seed_config=SEED_CONFIG)
    print(
        f"  Stopped at iteration {result.iterations}, best_score={result.best_score:.3f}"
    )


async def run_with_max_evaluations() -> None:
    """Stop after a fixed evaluation budget."""
    print("\n--- max_evaluations=15 ---")
    maw = MAW()
    opt = Optimizer(
        maw,
        criteria="Technical accuracy and clarity",
        config=OptimizationConfig(
            seed=42,
            minibatch_size=2,
            use_merge=False,
            max_iterations=20,
            max_evaluations=15,
            patience=20,
        ),
    )
    result = await opt.optimize(TRAINSET, seed_config=SEED_CONFIG)
    print(
        f"  Stopped at iteration {result.iterations}, "
        f"total_evals={result.total_evaluation_runs}, "
        f"best_score={result.best_score:.3f}"
    )


async def run_with_graceful_shutdown() -> None:
    """Show that graceful_shutdown option is available (SignalStopper).

    In practice, sending SIGINT/SIGTERM to this process would trigger
    graceful completion of the current iteration.
    """
    print("\n--- graceful_shutdown=True ---")
    maw = MAW()
    opt = Optimizer(
        maw,
        criteria="Technical accuracy and clarity",
        config=OptimizationConfig(
            seed=42,
            minibatch_size=2,
            use_merge=False,
            graceful_shutdown=True,
            max_iterations=3,
            patience=10,
        ),
    )
    result = await opt.optimize(TRAINSET, seed_config=SEED_CONFIG)
    print(f"  Completed {result.iterations} iterations with graceful shutdown enabled")
    print(f"  best_score={result.best_score:.3f}")


async def main() -> None:
    await run_with_patience()
    await run_with_score_threshold()
    await run_with_max_evaluations()
    await run_with_graceful_shutdown()


if __name__ == "__main__":
    asyncio.run(main())
