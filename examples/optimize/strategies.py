"""Example: comparing candidate selection strategies.

Runs the same optimization with 'pareto', 'best', and 'epsilon_greedy' selectors
to show how strategy choice affects exploration vs exploitation.
"""

import asyncio
from typing import Literal

from fedotmas import MAW, Optimizer
from fedotmas.maw.models import MAWAgentConfig, MAWConfig, MAWStepConfig
from fedotmas.optimize import OptimizationConfig

SEED_CONFIG = MAWConfig(
    agents=[
        MAWAgentConfig(
            name="analyst",
            model="openai/gpt-4o-mini",
            instruction="Analyze the given topic and list key points.",
            output_key="analysis",
        ),
        MAWAgentConfig(
            name="synthesizer",
            model="openai/gpt-4o-mini",
            instruction="Synthesize the analysis into a concise summary.",
            output_key="summary",
        ),
    ],
    pipeline=MAWStepConfig(
        type="sequential",
        children=[
            MAWStepConfig(type="agent", agent_name="analyst"),
            MAWStepConfig(type="agent", agent_name="synthesizer"),
        ],
    ),
)

TRAINSET = [
    "Compare Python and Rust for systems programming",
    "Evaluate pros and cons of microservices architecture",
    "Analyze trade-offs of eventual vs strong consistency",
]


async def run_with_strategy(
    strategy: Literal["pareto", "best", "epsilon_greedy"],
) -> None:
    maw = MAW()
    opt = Optimizer(
        maw,
        criteria="Technical accuracy, depth of analysis, and balanced perspective",
        config=OptimizationConfig(
            candidate_selection=strategy,
            seed=42,
            minibatch_size=2,
            use_merge=False,
            max_merge_attempts=0,
            max_iterations=5,
            patience=3,
        ),
    )
    result = await opt.optimize(TRAINSET, seed_config=SEED_CONFIG)
    print(
        f"  Strategy={strategy:<16} "
        f"best_score={result.best_score:.3f}  "
        f"candidates={len(result.all_candidates)}  "
        f"pareto_size={len(result.pareto_front())}"
    )


async def main() -> None:
    for strategy in ("pareto", "best", "epsilon_greedy"):
        await run_with_strategy(strategy)


if __name__ == "__main__":
    asyncio.run(main())
