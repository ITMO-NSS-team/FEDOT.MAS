from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fedotmas.common.logging import get_logger

_log = get_logger("fedotmas.plugins.checkpoint")

_WORKFLOW_PREFIXES = ("seq_", "par_", "loop_")


@dataclass(frozen=True)
class Checkpoint:
    """State snapshot taken at an agent boundary."""

    agent_name: str
    state: dict[str, Any]
    index: int


class CheckpointPlugin:
    """Snapshots session state after each agent completes.

    Implements :class:`MiddlewareProtocol`.
    """

    def __init__(self) -> None:
        self._checkpoints: list[Checkpoint] = []

    @property
    def checkpoints(self) -> list[Checkpoint]:
        return list(self._checkpoints)

    def get(self, agent_name: str) -> Checkpoint | None:
        """Last checkpoint for *agent_name*, or ``None``."""
        for cp in reversed(self._checkpoints):
            if cp.agent_name == agent_name:
                return cp
        return None

    def state_at(self, agent_name: str) -> dict[str, Any] | None:
        """State snapshot right after *agent_name* finished."""
        cp = self.get(agent_name)
        return dict(cp.state) if cp else None

    def clear(self) -> None:
        self._checkpoints.clear()

    async def before_agent(
        self, agent_name: str, state: dict[str, Any]
    ) -> dict[str, str] | None:
        return None

    async def after_agent(
        self, agent_name: str, state: dict[str, Any]
    ) -> None:
        if agent_name.startswith(_WORKFLOW_PREFIXES):
            return

        cp = Checkpoint(
            agent_name=agent_name,
            state=dict(state),
            index=len(self._checkpoints),
        )
        self._checkpoints.append(cp)
        _log.debug(
            "Checkpoint | agent={} index={} keys={}",
            cp.agent_name,
            cp.index,
            list(cp.state.keys()),
        )
