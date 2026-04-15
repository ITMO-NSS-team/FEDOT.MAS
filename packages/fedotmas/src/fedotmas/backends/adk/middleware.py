from __future__ import annotations

import asyncio
from typing import Optional

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.plugins import BasePlugin
from google.genai import types

from fedotmas.common.logging import get_logger
from fedotmas.interfaces.middleware import MiddlewareProtocol

_log = get_logger("fedotmas.backends.adk.middleware")

_WORKFLOW_PREFIXES = ("seq_", "par_", "loop_")


class MiddlewareAdapter(BasePlugin):
    """Wraps a list of :class:`MiddlewareProtocol` as an ADK ``BasePlugin``.

    Maps the minimal middleware hooks (before/after agent) to ADK's
    callback system.  Backend-specific plugins (LangfusePlugin, etc.)
    are passed separately and are not wrapped by this adapter.
    """

    def __init__(self, middlewares: list[MiddlewareProtocol]) -> None:
        super().__init__(name="fedotmas_middleware_adapter")
        self._middlewares = middlewares

    async def before_agent_callback(
        self, *, agent: BaseAgent, callback_context: CallbackContext
    ) -> Optional[types.Content]:
        if agent.name.startswith(_WORKFLOW_PREFIXES):
            return None

        state = dict(callback_context.state.to_dict())

        for mw in self._middlewares:
            result = await mw.before_agent(agent.name, state)
            if result is not None:
                # Short-circuit: middleware wants to skip this agent.
                text = result.get("text", f"[skipped {agent.name}]")
                return types.Content(
                    role="model",
                    parts=[types.Part.from_text(text=text)],
                )
        return None

    async def after_agent_callback(
        self, *, agent: BaseAgent, callback_context: CallbackContext
    ) -> Optional[types.Content]:
        if agent.name.startswith(_WORKFLOW_PREFIXES):
            return None

        state = dict(callback_context.state.to_dict())

        for mw in self._middlewares:
            await mw.after_agent(agent.name, state)

        return None
