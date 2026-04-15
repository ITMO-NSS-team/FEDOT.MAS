from __future__ import annotations

import time
from typing import Any

from fedotmas.common.logging import get_logger

_log = get_logger("fedotmas.plugins.logging")

_WORKFLOW_PREFIXES = ("seq_", "par_", "loop_")


def _is_workflow_node(name: str) -> bool:
    return name.startswith(_WORKFLOW_PREFIXES)


class LoggingPlugin:
    """Default FEDOT.MAS middleware that logs agent lifecycle.

    Implements :class:`MiddlewareProtocol`.
    """

    def __init__(self) -> None:
        self._agent_start: dict[str, float] = {}

    async def before_agent(
        self, agent_name: str, state: dict[str, Any]
    ) -> dict[str, str] | None:
        if not _is_workflow_node(agent_name):
            _log.info("Agent started | name={}", agent_name)
        self._agent_start[agent_name] = time.monotonic()
        return None

    async def after_agent(
        self, agent_name: str, state: dict[str, Any]
    ) -> None:
        t0 = self._agent_start.pop(agent_name, None)
        if t0 is not None and not _is_workflow_node(agent_name):
            _log.info(
                "Agent done | name={} elapsed={:.1f}s",
                agent_name,
                time.monotonic() - t0,
            )
