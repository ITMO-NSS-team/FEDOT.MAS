"""Example: separate training and validation sets.

Demonstrates how to use a distinct valset to prevent overfitting.
The optimizer mutates on minibatches from trainset, and accepted candidates
are evaluated on the full valset before joining the Pareto front.
"""

import asyncio

from fedotmas import MAW, Optimizer
from fedotmas.maw.models import MAWAgentConfig, MAWConfig, MAWStepConfig
from fedotmas.optimize import OptimizationConfig

SEED_CONFIG = MAWConfig(
    agents=[
        MAWAgentConfig(
            name="researcher",
            model="openai/gpt-4o-mini",
            instruction="Research the given topic thoroughly and provide key findings.",
            output_key="research",
        ),
        MAWAgentConfig(
            name="writer",
            model="openai/gpt-4o-mini",
            instruction="Write a structured summary based on the research findings.",
            output_key="summary",
        ),
    ],
    pipeline=MAWStepConfig(
        type="sequential",
        children=[
            MAWStepConfig(type="agent", agent_name="researcher"),
            MAWStepConfig(type="agent", agent_name="writer"),
        ],
    ),
)

# Training tasks: used for minibatch mutation and reflection
TRAINSET = [
    "Explain the impact of quantum computing on cryptography",
    "Analyze the pros and cons of nuclear energy",
    "Discuss the role of microbiomes in human health",
    "Compare REST and GraphQL API design approaches",
    "Evaluate the effectiveness of carbon offset programs",
]

# Validation tasks: used to score accepted candidates on unseen tasks
VALSET = [
    "Assess the future of autonomous vehicles in urban areas",
    "Analyze the economic impact of open-source software",
    "Discuss ethical implications of gene editing in humans",
]


async def main() -> None:
    maw = MAW()

    opt = Optimizer(
        maw,
        criteria="Factual accuracy, depth of analysis, clear structure, and balanced perspective",
        config=OptimizationConfig(
            max_iterations=6,
            patience=4,
            minibatch_size=2,
            seed=42,
        ),
    )

    result = await opt.optimize(
        TRAINSET,
        seed_config=SEED_CONFIG,
        valset=VALSET,  # separate validation set
    )

    print(f"Best score: {result.best_score:.3f}")
    print(f"Iterations: {result.iterations}")
    print(f"Candidates: {len(result.all_candidates)}")
    print(f"Pareto front: {len(result.pareto_front())}")

    # Show per-task scores of the best candidate
    best = max(result.all_candidates, key=lambda c: c.mean_score or 0.0)
    print("\n--- Best candidate scores ---")
    for task, score in sorted(best.scores.items()):
        source = "train" if task in TRAINSET else "val"
        print(f"  [{source}] {task[:50]}... = {score:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
