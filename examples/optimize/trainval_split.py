import asyncio

from fedotmas import MAW, Optimizer
from fedotmas.maw.models import MAWAgentConfig, MAWConfig, MAWStepConfig
from fedotmas.optimize import OptimizationConfig, Task

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

TRAINSET = [
    Task("Explain the impact of quantum computing on cryptography"),
    Task("Analyze the pros and cons of nuclear energy"),
    Task("Discuss the role of microbiomes in human health"),
    Task("Compare REST and GraphQL API design approaches"),
    Task("Evaluate the effectiveness of carbon offset programs"),
]

VALSET = [
    Task("Assess the future of autonomous vehicles in urban areas"),
    Task("Analyze the economic impact of open-source software"),
    Task("Discuss ethical implications of gene editing in humans"),
]


async def main() -> None:
    maw = MAW()

    opt = Optimizer(
        maw,
        criteria="Factual accuracy, depth of analysis, clear structure, and balanced perspective",
        config=OptimizationConfig(
            max_iterations=3,
            patience=4,
            minibatch_size=2,
            seed=42,
        ),
    )

    result = await opt.optimize(
        TRAINSET,
        seed_config=SEED_CONFIG,
        valset=VALSET,
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
