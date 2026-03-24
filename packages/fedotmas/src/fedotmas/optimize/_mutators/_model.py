from __future__ import annotations

from fedotmas.maw.models import MAWConfig
from fedotmas.optimize._state import Candidate


class ModelMutator:
    """Optimize model selection per agent."""

    @property
    def token_usage(self) -> tuple[int, int]:
        return (0, 0)

    async def mutate(
        self,
        candidate: Candidate,
        agent_names: list[str],
        tasks: list[str],
    ) -> MAWConfig:
        raise NotImplementedError("ModelMutator is not yet implemented")

    async def merge(
        self,
        candidate_a: Candidate,
        candidate_b: Candidate,
        tasks: list[str],
    ) -> MAWConfig:
        raise NotImplementedError("ModelMutator is not yet implemented")

    async def genealogy_merge(
        self,
        ancestor: Candidate,
        child_a: Candidate,
        child_b: Candidate,
        tasks: list[str],
    ) -> MAWConfig:
        raise NotImplementedError("ModelMutator is not yet implemented")
