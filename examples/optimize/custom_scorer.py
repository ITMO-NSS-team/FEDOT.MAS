"""Example: custom Scorer using deterministic regex/keyword matching.

Demonstrates that Optimizer accepts any object implementing the Scorer protocol
(async evaluate(task, state) -> ScoringResult).
"""

import asyncio
import re

from fedotmas import MAW, Optimizer
from fedotmas.maw.models import MAWAgentConfig, MAWConfig, MAWStepConfig
from fedotmas.optimize import OptimizationConfig, ScoringResult


class KeywordScorer:
    """Deterministic scorer that checks for required keywords in pipeline output."""

    KEYWORDS_BY_TASK = {
        "Summarize benefits of solar energy": [
            "renewable",
            "cost",
            "environment",
            "efficiency",
        ],
        "Explain how batteries store energy": [
            "chemical",
            "lithium",
            "discharge",
            "capacity",
        ],
        "Describe wind turbine components": [
            "blade",
            "generator",
            "tower",
            "rotor",
        ],
    }

    async def evaluate(self, task: str, state: dict) -> ScoringResult:
        keywords = self.KEYWORDS_BY_TASK.get(task, [])
        if not keywords:
            return ScoringResult(
                score=0.5, feedback="No keywords defined", reasoning=""
            )

        output_text = " ".join(str(v) for v in state.values()).lower()
        found = [kw for kw in keywords if re.search(rf"\b{kw}\b", output_text)]
        score = len(found) / len(keywords)
        missing = [kw for kw in keywords if kw not in found]

        feedback = f"Found {len(found)}/{len(keywords)} keywords." + (
            f" Missing: {', '.join(missing)}." if missing else " All keywords present."
        )
        return ScoringResult(score=score, feedback=feedback, reasoning=feedback)


SEED_CONFIG = MAWConfig(
    agents=[
        MAWAgentConfig(
            name="researcher",
            instruction="Research the given topic and produce a comprehensive summary.",
            output_key="research",
        ),
        MAWAgentConfig(
            name="writer",
            instruction="Write a clear, detailed explanation based on the research.",
            output_key="article",
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


async def main() -> None:
    maw = MAW()

    trainset = list(KeywordScorer.KEYWORDS_BY_TASK.keys())

    opt = Optimizer(
        maw,
        scorer=KeywordScorer(),
        config=OptimizationConfig(
            seed=42,
            minibatch_size=2,
            use_merge=False,
            max_iterations=5,
            patience=3,
        ),
    )

    result = await opt.optimize(trainset, seed_config=SEED_CONFIG)

    print(f"Best score: {result.best_score:.3f}")
    print(f"Candidates evaluated: {len(result.all_candidates)}")
    print(f"Iterations: {result.iterations}")


if __name__ == "__main__":
    asyncio.run(main())
