from __future__ import annotations

from typing import Any

from fedotmas.common.logging import get_logger

_log = get_logger("fedotmas.plugins.skip_completed")


class SkipCompletedPlugin:
    """Skips agents whose names are in the completed set.

    Implements :class:`MiddlewareProtocol`.

    Used by ``Controller.resume`` to avoid re-executing agents
    whose output is already available from a previous checkpoint.
    """

    def __init__(self, completed_agents: set[str]) -> None:
        self._completed = frozenset(completed_agents)

    async def before_agent(
        self, agent_name: str, state: dict[str, Any]
    ) -> dict[str, str] | None:
        if agent_name in self._completed:
            _log.debug("Skipping completed agent: {}", agent_name)
            return {"text": f"[skipped {agent_name}]"}
        return None

    async def after_agent(
        self, agent_name: str, state: dict[str, Any]
    ) -> None:
        pass
