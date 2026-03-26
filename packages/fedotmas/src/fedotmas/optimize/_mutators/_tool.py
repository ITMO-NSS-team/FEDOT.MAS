from __future__ import annotations

from fedotmas.maw.models import MAWConfig
from fedotmas.optimize._state import Candidate, Task


class ToolMutator:
    """Optimize agent tool assignments (add/remove/replace tools per agent)."""

    @property
    def token_usage(self) -> tuple[int, int]:
        return (0, 0)

    async def mutate(
        self,
        candidate: Candidate,
        agent_names: list[str],
        tasks: list[Task],
    ) -> MAWConfig:
        raise NotImplementedError("ToolMutator is not yet implemented")

    async def merge(
        self,
        candidate_a: Candidate,
        candidate_b: Candidate,
        tasks: list[Task],
    ) -> MAWConfig:
        raise NotImplementedError("ToolMutator is not yet implemented")

    async def genealogy_merge(
        self,
        ancestor: Candidate,
        child_a: Candidate,
        child_b: Candidate,
        tasks: list[Task],
    ) -> MAWConfig:
        raise NotImplementedError("ToolMutator is not yet implemented")
