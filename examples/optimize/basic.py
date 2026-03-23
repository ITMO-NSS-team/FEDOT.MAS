import asyncio

from fedotmas import MAW, MAWConfig, Optimizer
from fedotmas.maw.models import MAWAgentConfig, MAWStepConfig
from fedotmas.optimize import OptimizationConfig

SEED_CONFIG = MAWConfig(
    agents=[
        MAWAgentConfig(
            name="planner",
            model="openai/gpt-4o-mini",
            instruction="Create a structured plan for the given task.",
            output_key="plan",
        ),
        MAWAgentConfig(
            name="executor",
            model="openai/gpt-4o-mini",
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


async def main() -> None:
    maw = MAW()

    # Define training tasks
    trainset = [
        "Analyze the current state of renewable energy adoption globally",
        "Compare cloud computing providers for a startup",
        "Research best practices for remote team management",
    ]

    # Create optimizer with evaluation criteria
    opt = Optimizer(
        maw,
        criteria="Completeness, accuracy, and actionability of the analysis",
        config=OptimizationConfig(
            max_iterations=5,
            patience=3,
            max_merge_attempts=3,
            minibatch_size=2,
        ),
    )

    # Run optimization
    result = await opt.optimize(trainset, seed_config=SEED_CONFIG)

    print(f"Best score: {result.best_score:.3f}")
    print(f"Iterations: {result.iterations}")
    print(f"Candidates evaluated: {len(result.all_candidates)}")
    print(f"Pareto front size: {len(result.pareto_front())}")
    print(f"Total evaluation runs: {result.total_evaluation_runs}")
    print(
        f"Tokens: {result.total_prompt_tokens} prompt / {result.total_completion_tokens} completion"
    )
    print(f"\nBest config:\n{result.best_config}")


if __name__ == "__main__":
    asyncio.run(main())
