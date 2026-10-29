from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.plugins import BasePlugin
from google.genai import types

from fedotmas.common.logging import get_logger

_log = get_logger("fedotmas.plugins.checkpoint")

_WORKFLOW_PREFIXES = ("seq_", "par_", "loop_")


@dataclass(frozen=True)
class Checkpoint:
    """State snapshot taken at an agent boundary."""

    agent_name: str
    state: dict[str, Any]
    index: int
    validated: bool = True


class CheckpointPlugin(BasePlugin):
    """Snapshots session state after each agent completes.

    Checkpoints are stored in-memory and can be used for inspection,
    rewind, or retry from a specific pipeline stage.
    """

    def __init__(self, checkpoints: list[Checkpoint] | None = None) -> None:
        super().__init__(name="fedotmas_checkpoint")
        # ADK does not call after_agent for skipped agents. Carry their original
        # boundaries forward so a second resume can still reuse that prefix.
        self._checkpoints: list[Checkpoint] = deepcopy(checkpoints or [])

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
        return deepcopy(cp.state) if cp else None

    def clear(self) -> None:
        self._checkpoints.clear()

    async def after_agent_callback(
        self, *, agent: BaseAgent, callback_context: CallbackContext
    ) -> types.Content | None:
        if agent.name.startswith(_WORKFLOW_PREFIXES):
            return None

        cp = Checkpoint(
            agent_name=agent.name,
            state=deepcopy(callback_context.state.to_dict()),
            index=len(self._checkpoints),
            validated=False,
        )
        self._checkpoints.append(cp)
        # Plugin callbacks run before the agent's contract/source checks. Force
        # a completion event and finalize the snapshot only after those checks.
        callback_context.state["__fedotmas_checkpoint_ready"] = cp.index
        _log.debug(
            "Checkpoint | agent={} index={} keys={}",
            cp.agent_name,
            cp.index,
            list(cp.state.keys()),
        )
        return None

    async def on_event_callback(self, *, invocation_context, event):
        delta = event.actions.state_delta if event.actions else {}
        index = delta.get("__fedotmas_checkpoint_ready")
        if type(index) is not int or not 0 <= index < len(self._checkpoints):
            return None
        pending = self._checkpoints[index]
        if pending.agent_name != event.author or pending.validated:
            return None
        state = deepcopy(dict(invocation_context.session.state))
        state.update(deepcopy(delta))
        self._checkpoints[index] = Checkpoint(
            pending.agent_name, state, index, validated=True
        )
        return None
