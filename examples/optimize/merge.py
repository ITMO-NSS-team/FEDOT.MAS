"""Example: merge pipeline and Pareto front.

Demonstrates genealogy-aware merge (when a common ancestor exists) with
LLM-merge fallback, and prints the Pareto front with candidate origins.
"""

import asyncio

from fedotmas import MAW, Optimizer
from fedotmas.maw.models import MAWAgentConfig, MAWConfig, MAWStepConfig
from fedotmas.optimize import OptimizationCallback, OptimizationConfig
from fedotmas.optimize._state import Candidate

SEED_CONFIG = MAWConfig(
    agents=[
        MAWAgentConfig(
            name="analyst",
            model="openai/gpt-4o-mini",
            instruction="Analyze the given topic and identify key factors.",
            output_key="analysis",
        ),
        MAWAgentConfig(
            name="critic",
            model="openai/gpt-4o-mini",
            instruction="Critically evaluate the analysis and suggest improvements.",
            output_key="critique",
        ),
        MAWAgentConfig(
            name="writer",
            model="openai/gpt-4o-mini",
            instruction="Produce a final report incorporating analysis and critique.",
            output_key="report",
        ),
    ],
    pipeline=MAWStepConfig(
        type="sequential",
        children=[
            MAWStepConfig(type="agent", agent_name="analyst"),
            MAWStepConfig(type="agent", agent_name="critic"),
            MAWStepConfig(type="agent", agent_name="writer"),
        ],
    ),
)

TRAINSET = [
    "Evaluate the impact of AI on healthcare diagnostics",
    "Analyze the economic effects of remote work",
    "Assess cybersecurity risks in IoT devices",
    "Compare electric vs hydrogen fuel cell vehicles",
]


class MergeTracker(OptimizationCallback):
    def __init__(self) -> None:
        self.merges: list[tuple[int, int]] = []

    def on_merge_attempted(self, pair: tuple[Candidate, Candidate]) -> None:
        self.merges.append((pair[0].index, pair[1].index))


async def main() -> None:
    maw = MAW()
    tracker = MergeTracker()

    opt = Optimizer(
        maw,
        criteria="Depth of analysis, balanced perspective, actionable conclusions",
        config=OptimizationConfig(
            seed=42,
            use_merge=True,
            max_merge_attempts=5,
            minibatch_size=2,
            candidate_selection="pareto",
            max_iterations=8,
            patience=5,
        ),
        callbacks=[tracker],
    )

    result = await opt.optimize(TRAINSET, seed_config=SEED_CONFIG)

    print(f"Best score: {result.best_score:.3f}")
    print(f"Candidates: {len(result.all_candidates)}")
    print(f"Merge attempts: {len(tracker.merges)}")

    print("\n--- Pareto Front ---")
    for c in result.pareto_front():
        merge_info = ""
        if c.merge_parent_indices is not None:
            merge_info = f"  merge_parents={c.merge_parent_indices}"
        print(
            f"  #{c.index}  origin={c.origin:<10}  "
            f"parent={c.parent_index}  "
            f"mean_score={c.mean_score or 0:.3f}"
            f"{merge_info}"
        )

    print("\n--- Candidate Genealogy ---")
    for c in result.all_candidates:
        merge_info = ""
        if c.merge_parent_indices is not None:
            merge_info = f"  merge_parents={c.merge_parent_indices}"
        print(
            f"  #{c.index}  origin={c.origin:<10}  "
            f"parent={c.parent_index}  "
            f"scored={bool(c.scores)}"
            f"{merge_info}"
        )


if __name__ == "__main__":
    asyncio.run(main())
