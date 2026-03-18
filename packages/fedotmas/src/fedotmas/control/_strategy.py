from __future__ import annotations

from enum import Enum
from typing import Any

from fedotmas.plugins._checkpoint import Checkpoint


class Strategy(str, Enum):
    RETRY_FAILED = "retry_failed"
    RESTART_AFTER = "restart_after"
    RESTART_ALL = "restart_all"


def resolve_initial_state(
    strategy: Strategy,
    checkpoints: list[Checkpoint],
    old_config: object,
    new_config: object,
) -> tuple[dict[str, Any] | None, set[str]]:
    """Determine initial state and completed agents for a resume run.

    Returns:
        (initial_state, completed_agent_names)
    """
    match strategy:
        case Strategy.RESTART_ALL:
            return None, set()

        case _ if not checkpoints:
            return None, set()

        case Strategy.RETRY_FAILED:
            completed = {cp.agent_name for cp in checkpoints}
            return dict(checkpoints[-1].state), completed

        case Strategy.RESTART_AFTER:
            return _resolve_restart_after(checkpoints, old_config, new_config)


def _resolve_restart_after(
    checkpoints: list[Checkpoint],
    old_config: object,
    new_config: object,
) -> tuple[dict[str, Any] | None, set[str]]:
    from fedotmas.maw.models import MAWConfig

    if not isinstance(old_config, MAWConfig) or not isinstance(new_config, MAWConfig):
        return None, set()

    old_agents = {a.name: a for a in old_config.agents}
    new_agents = {a.name: a for a in new_config.agents}

    cutoff = len(checkpoints)
    for i, cp in enumerate(checkpoints):
        if cp.agent_name not in new_agents:
            cutoff = i
            break
        if old_agents.get(cp.agent_name) != new_agents.get(cp.agent_name):
            cutoff = i
            break

    if cutoff == 0:
        return None, set()

    kept = checkpoints[:cutoff]
    return dict(kept[-1].state), {cp.agent_name for cp in kept}
