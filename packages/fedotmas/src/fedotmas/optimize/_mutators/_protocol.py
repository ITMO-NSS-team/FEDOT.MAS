from __future__ import annotations

from typing import Protocol, runtime_checkable

from fedotmas.maw.models import MAWConfig
from fedotmas.optimize._state import Candidate, Task


@runtime_checkable
class Mutator(Protocol):
    """Protocol for mutation operators.

    Each mutator modifies one aspect of the agent graph configuration
    (instructions, tools, models, structure).
    """

    async def mutate(
        self,
        candidate: Candidate,
        agent_names: list[str],
        tasks: list[Task],
    ) -> MAWConfig:
        """Produce a mutated config from the given candidate.

        Args:
            candidate: Parent candidate to mutate.
            agent_names: Which agents to mutate this iteration.
            tasks: Current task batch for reflection context.

        Returns:
            New MAWConfig with mutations applied.
        """
        ...

    async def merge(
        self,
        candidate_a: Candidate,
        candidate_b: Candidate,
        tasks: list[Task],
    ) -> MAWConfig:
        """Merge two candidates into one.

        Args:
            candidate_a: First parent.
            candidate_b: Second parent.
            tasks: Task context for merge decisions.

        Returns:
            Merged MAWConfig.
        """
        ...

    async def genealogy_merge(
        self,
        ancestor: Candidate,
        child_a: Candidate,
        child_b: Candidate,
        tasks: list[Task],
    ) -> MAWConfig:
        """Merge using common ancestor for smarter crossover."""
        ...

    @property
    def token_usage(self) -> tuple[int, int]:
        """Return (prompt_tokens, completion_tokens) consumed."""
        ...
