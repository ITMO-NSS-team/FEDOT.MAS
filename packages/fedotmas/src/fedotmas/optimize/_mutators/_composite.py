from __future__ import annotations

import random
from dataclasses import dataclass

from fedotmas.maw.models import MAWConfig
from fedotmas.optimize._mutators._protocol import Mutator
from fedotmas.optimize._state import Candidate


@dataclass(frozen=True)
class WeightedMutator:
    mutator: Mutator
    weight: float = 1.0


class CompositeMutator:
    """Composes multiple mutators with weighted random selection.

    - mutate(): picks ONE mutator by weight (one change per iteration).
    - merge/genealogy_merge(): applies ALL mutators sequentially.
    - token_usage: aggregated across all mutators.
    """

    def __init__(
        self,
        mutators: list[WeightedMutator],
        rng: random.Random | None = None,
    ) -> None:
        if not mutators:
            raise ValueError("At least one mutator is required")
        self._mutators = mutators
        self._rng = rng or random.Random()

    async def mutate(
        self,
        candidate: Candidate,
        agent_names: list[str],
        tasks: list[str],
    ) -> MAWConfig:
        chosen = self._weighted_choice()
        return await chosen.mutator.mutate(candidate, agent_names, tasks)

    async def merge(
        self,
        candidate_a: Candidate,
        candidate_b: Candidate,
        tasks: list[str],
    ) -> MAWConfig:
        config = candidate_a.config
        for wm in self._mutators:
            config = await wm.mutator.merge(candidate_a, candidate_b, tasks)
        return config

    async def genealogy_merge(
        self,
        ancestor: Candidate,
        child_a: Candidate,
        child_b: Candidate,
        tasks: list[str],
    ) -> MAWConfig:
        config = child_a.config
        for wm in self._mutators:
            config = await wm.mutator.genealogy_merge(ancestor, child_a, child_b, tasks)
        return config

    @property
    def token_usage(self) -> tuple[int, int]:
        p = sum(wm.mutator.token_usage[0] for wm in self._mutators)
        c = sum(wm.mutator.token_usage[1] for wm in self._mutators)
        return (p, c)

    def _weighted_choice(self) -> WeightedMutator:
        weights = [wm.weight for wm in self._mutators]
        return self._rng.choices(self._mutators, weights=weights, k=1)[0]
