from __future__ import annotations

from typing import Any, Optional

from google.adk.plugins import BasePlugin
from google.adk.runners import InvocationContext
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext

from fedotmas.common.logging import get_logger

_log = get_logger("fedotmas.plugins.web_search_limit")

DEFAULT_WEB_SEARCH_TOOL_NAMES = frozenset(
    {
        "search",
        "web_search",
        "web-search",
        "websearch",
        "google_search",
        "searxng_search",
    }
)
WEB_SEARCH_HINTS = (
    "web",
    "internet",
    "searxng",
    "google",
    "bing",
    "duckduckgo",
    "brave",
    "yahoo",
)


class WebSearchLimitPlugin(BasePlugin):
    """Limit web-search tool calls per agent within one ADK run."""

    def __init__(
        self,
        *,
        max_calls_per_agent: int = 4,
        tool_names: set[str] | None = None,
    ) -> None:
        if max_calls_per_agent < 1:
            raise ValueError("max_calls_per_agent must be >= 1")
        super().__init__(name="fedotmas_web_search_limit")
        self.max_calls_per_agent = max_calls_per_agent
        self._tool_names = {
            name.lower() for name in (tool_names or DEFAULT_WEB_SEARCH_TOOL_NAMES)
        }
        self._counts: dict[tuple[str, str], int] = {}

    async def before_run_callback(
        self, *, invocation_context: InvocationContext
    ) -> None:
        session_id = invocation_context.session.id
        self._counts = {
            key: count for key, count in self._counts.items() if key[0] != session_id
        }
        return None

    async def before_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
    ) -> Optional[dict]:
        if not self._is_web_search_tool(tool):
            return None

        session_id = tool_context._invocation_context.session.id
        agent_name = tool_context._invocation_context.agent.name
        key = (session_id, agent_name)
        used = self._counts.get(key, 0)
        if used >= self.max_calls_per_agent:
            message = (
                "Web search limit exceeded for agent "
                f"'{agent_name}': max {self.max_calls_per_agent} calls per run."
            )
            _log.warning(message)
            return {"isError": True, "error": message}

        self._counts[key] = used + 1
        _log.debug(
            "Web search call allowed | agent={} tool={} used={}/{}",
            agent_name,
            tool.name,
            used + 1,
            self.max_calls_per_agent,
        )
        return None

    def _is_web_search_tool(self, tool: BaseTool) -> bool:
        name = tool.name.lower()
        if name in self._tool_names:
            description = (tool.description or "").lower()
            if name == "search":
                return any(hint in description for hint in WEB_SEARCH_HINTS)
            return True
        return False
