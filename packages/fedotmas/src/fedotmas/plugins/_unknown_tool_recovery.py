from __future__ import annotations

import re
from typing import Any, Optional

from google.adk.plugins import BasePlugin
from google.adk.runners import InvocationContext
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext

from fedotmas.common.logging import get_logger

_log = get_logger("fedotmas.plugins.unknown_tool_recovery")

#: ADK's message for a function call naming something absent from the agent's
#: tools.  Deliberately anchored: a tool of our own raising "file not found"
#: must not be mistaken for an unresolved tool name.
_UNKNOWN_TOOL_MESSAGE = re.compile(r"^Tool '[^']*' not found\.")


class UnknownToolRecoveryPlugin(BasePlugin):
    """Keep a hallucinated tool name from aborting the whole pipeline.

    ADK raises when a model calls a name absent from its tools, which discards
    the output of every step that already ran.  This hands the model an
    ordinary error result instead, so it can pick a real tool.

    Recovery is bounded: after *max_recoveries_per_agent* unresolved names the
    agent is clearly not correcting itself, and the error is left to propagate
    as it did before.
    """

    def __init__(
        self,
        *,
        max_recoveries_per_agent: int = 3,
        name: str = "fedotmas_unknown_tool_recovery",
    ) -> None:
        if max_recoveries_per_agent < 1:
            raise ValueError("max_recoveries_per_agent must be >= 1")
        super().__init__(name=name)
        self.max_recoveries_per_agent = max_recoveries_per_agent
        self._recoveries: dict[tuple[str, str], int] = {}

    async def before_run_callback(
        self, *, invocation_context: InvocationContext
    ) -> None:
        # Only this session's, never the whole dict: clearing would hand a
        # still-running sibling a fresh budget, so the bound would never fire.
        session_id = invocation_context.session.id
        self._recoveries = {
            key: count
            for key, count in self._recoveries.items()
            if key[0] != session_id
        }
        return None

    async def on_tool_error_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        error: Exception,
    ) -> Optional[dict]:
        if not _is_unknown_tool_error(tool, error):
            return None

        # Bound to a local so the suppression stays on its own short line: the
        # formatter wraps a longer one and leaves the comment on the closing
        # paren, where ty stops honouring it and the check turns red.
        invocation = tool_context._invocation_context
        session_id = invocation.session.id
        agent_name = invocation.agent.name  # ty: ignore[unresolved-attribute]
        key = (session_id, agent_name)
        used = self._recoveries.get(key, 0) + 1
        self._recoveries[key] = used

        if used > self.max_recoveries_per_agent:
            _log.warning(
                "Unknown tool '{}' called by {} after {} recoveries; giving up",
                tool.name,
                agent_name,
                self.max_recoveries_per_agent,
            )
            return None

        _log.warning(
            "Unknown tool '{}' called by {}; reporting back to the model ({}/{})",
            tool.name,
            agent_name,
            used,
            self.max_recoveries_per_agent,
        )
        return {
            # MCP's error shape, which LoggingPlugin keys on; without it a
            # recovered call is logged as an ordinary result.
            "isError": True,
            "error": f"Tool '{tool.name}' does not exist.",
            "hint": (
                "Call only the tools that were provided to you. If none of them "
                "fits, answer with the information you already have."
            ),
        }


def _is_unknown_tool_error(tool: BaseTool, error: Exception) -> bool:
    """Whether ADK failed to resolve the name the model asked for.

    When a function call names something absent from the agent's tools, ADK
    raises ``ValueError`` and passes this callback a bare ``BaseTool`` stub
    rather than a real tool.  Two independent signals identify that, because
    both are ADK internals and either could move: the stub's exact type (every
    genuine tool is a ``BaseTool`` *subclass*) and the message shape.  Being
    too strict here merely restores the old crash; being too lax would swallow
    real tool failures, so each signal is narrow on its own.
    """
    if not isinstance(error, ValueError):
        return False
    return type(tool) is BaseTool or bool(_UNKNOWN_TOOL_MESSAGE.match(str(error)))
