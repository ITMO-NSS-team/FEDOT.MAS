from __future__ import annotations

from typing import Optional

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.plugins import BasePlugin
from google.genai import types

from fedotmas.common.logging import get_logger

_log = get_logger("fedotmas.plugins.skip_completed")

_WORKFLOW_PREFIXES = ("seq_", "par_", "loop_")


class SkipCompletedPlugin(BasePlugin):
    """Skips agents whose names are in the completed set.

    Used by ``Controller.resume`` to avoid re-executing agents
    whose output is already available from a previous checkpoint.
    """

    def __init__(self, completed_agents: set[str]) -> None:
        super().__init__(name="fedotmas_skip_completed")
        self._completed = frozenset(completed_agents)

    async def before_agent_callback(
        self, *, agent: BaseAgent, callback_context: CallbackContext
    ) -> Optional[types.Content]:
        if agent.name.startswith(_WORKFLOW_PREFIXES):
            return None

        if agent.name in self._completed:
            _log.debug("Skipping completed agent: {}", agent.name)
            return types.Content(
                role="model",
                parts=[types.Part.from_text(text=f"[skipped {agent.name}]")],
            )
        return None
