"""Example: checkpoint and resume.

Runs optimization for 3 iterations, saves state to a checkpoint file,
then resumes from the same checkpoint to verify state restoration.
"""

import asyncio
import tempfile
from pathlib import Path

from fedotmas import MAW, Optimizer
from fedotmas.maw.models import MAWAgentConfig, MAWConfig, MAWStepConfig
from fedotmas.optimize import OptimizationConfig, Task

SEED_CONFIG = MAWConfig(
    agents=[
        MAWAgentConfig(
            name="researcher",
            model="openai/gpt-4o-mini",
            instruction="Research the topic thoroughly.",
            output_key="research",
        ),
        MAWAgentConfig(
            name="writer",
            model="openai/gpt-4o-mini",
            instruction="Write a clear report based on the research.",
            output_key="report",
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
    Task("Explain quantum computing basics"),
    Task("Describe how neural networks learn"),
    Task("Summarize the history of the internet"),
]


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint = str(Path(tmpdir) / "optimizer_state.json")

        maw = MAW()

        # Phase 1: run 3 iterations
        print("--- Phase 1: initial run (3 iterations) ---")
        opt1 = Optimizer(
            maw,
            criteria="Clarity and technical accuracy",
            config=OptimizationConfig(
                seed=42,
                minibatch_size=2,
                checkpoint_path=checkpoint,
                use_merge=False,
                max_iterations=3,
                patience=10,
            ),
        )
        result1 = await opt1.optimize(TRAINSET, seed_config=SEED_CONFIG)
        print(
            f"Phase 1 done: iterations={result1.iterations} candidates={len(result1.all_candidates)}"
        )
        print(f"Best score: {result1.best_score:.3f}")

        assert Path(checkpoint).exists(), "Checkpoint file should exist"

        # Phase 2: resume and run 3 more
        print("\n--- Phase 2: resume (3 more iterations) ---")
        opt2 = Optimizer(
            maw,
            criteria="Clarity and technical accuracy",
            config=OptimizationConfig(
                seed=42,
                minibatch_size=2,
                checkpoint_path=checkpoint,
                use_merge=False,
                max_iterations=6,
                patience=10,
            ),
        )
        result2 = await opt2.optimize(TRAINSET, seed_config=SEED_CONFIG)
        print(
            f"Phase 2 done: iterations={result2.iterations} candidates={len(result2.all_candidates)}"
        )
        print(f"Best score: {result2.best_score:.3f}")

        assert (
            result2.iterations >= result1.iterations
        ), "Should have more iterations after resume"
        assert len(result2.all_candidates) >= len(
            result1.all_candidates
        ), "Should have at least as many candidates after resume"
        print("\nCheckpoint/resume verified successfully.")


if __name__ == "__main__":
    asyncio.run(main())
