from __future__ import annotations

import warnings
from typing import Any
from urllib.parse import urldefrag, urlsplit, urlunsplit

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.plugins import BasePlugin
from google.adk.runners import InvocationContext
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext

from fedotmas.common.logging import get_logger
from fedotmas.mcp import strip_tool_name_prefix
from fedotmas.mcp.capabilities import (
    ToolCapability,
    is_inspection_tool,
    normalize_tool_name,
    tool_capability,
)
from fedotmas.plugins._research_telemetry import ResearchTelemetry
from fedotmas.plugins._tool_error_circuit_breaker import (
    DUPLICATE_TOOL_CALL,
    WEB_BUDGET_EXHAUSTED,
)

_log = get_logger("fedotmas.plugins.web_search_limit")

_WEB_SEARCH_DESCRIPTION_HINTS = (
    "web",
    "internet",
    "searxng",
    "google",
    "bing",
    "duckduckgo",
    "brave",
    "yahoo",
    "tavily",
)

BUDGET_STATE_KEY = "_fedotmas_tool_budgets"


#: Searches per agent before the budget answers "stop exploring".  Four ran out
#: on every research task in live runs; the GAIA runner sets its own limits.
DEFAULT_WEB_SEARCH_LIMIT = 20


class WebSearchLimitPlugin(BasePlugin):
    """Limit web-search calls per agent within one ADK run.

    Exhaustion returns a marked tool result for that agent to finalize from
    current evidence. The legacy ``hard_fail=True`` argument warns and is ignored:
    raising here aborts ADK's whole parallel tool-call batch.
    """

    def __init__(
        self,
        *,
        max_calls_per_agent: int = DEFAULT_WEB_SEARCH_LIMIT,
        tool_names: set[str] | None = None,
        count_unique_urls: bool = False,
        same_url_exempt_tool_names: set[str] | None = None,
        ignore_local_urls: bool = True,
        reject_empty_urls: bool = False,
        dedupe_identical_calls: bool = True,
        hard_fail: bool | None = None,
        exhausted_agents: set[tuple[str, str, str]] | None = None,
        telemetry: ResearchTelemetry | None = None,
        budget_kind: str = "search",
        name: str = "fedotmas_web_search_limit",
    ) -> None:
        if max_calls_per_agent < 1:
            raise ValueError("max_calls_per_agent must be >= 1")
        super().__init__(name=name)
        self.max_calls_per_agent = max_calls_per_agent
        if hard_fail:
            warnings.warn(
                "hard_fail=True is deprecated and ignored; budget exhaustion is "
                "always returned as a tool result so parallel calls stay correlated.",
                DeprecationWarning,
                stacklevel=2,
            )
        self.count_unique_urls = count_unique_urls
        self.ignore_local_urls = ignore_local_urls
        self.reject_empty_urls = reject_empty_urls
        self.dedupe_identical_calls = dedupe_identical_calls
        self.telemetry = telemetry
        self.budget_kind = budget_kind
        self._tool_names = {name.lower() for name in (tool_names or set())}
        self._custom_tool_names = tool_names is not None
        self._same_url_exempt_tool_names = {
            name.lower() for name in (same_url_exempt_tool_names or set())
        }
        self._counts: dict[tuple[str, str], int] = {}
        self._seen_calls: set[tuple[str, str, str]] = set()
        self._seen_urls: set[tuple[str, str, str]] = set()
        self._exhausted_agents = (
            exhausted_agents if exhausted_agents is not None else set()
        )

    async def before_run_callback(
        self, *, invocation_context: InvocationContext
    ) -> None:
        session_id = invocation_context.session.id
        self._counts = {
            key: count for key, count in self._counts.items() if key[0] != session_id
        }
        self._seen_calls = {key for key in self._seen_calls if key[0] != session_id}
        self._seen_urls = {key for key in self._seen_urls if key[0] != session_id}
        stale_agents = {key for key in self._exhausted_agents if key[0] == session_id}
        self._exhausted_agents.difference_update(stale_agents)

    async def before_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
    ) -> dict | None:
        if not self._is_web_search_tool(tool):
            return None

        session_id = tool_context._invocation_context.session.id
        agent_name = tool_context._invocation_context.agent.name  # ty: ignore[unresolved-attribute]
        tool_name = strip_tool_name_prefix(tool.name).lower()
        key = (session_id, agent_name, self.budget_kind)
        state = getattr(tool_context, "state", None)
        state_writable = all(hasattr(state, attr) for attr in ("get", "__setitem__"))
        budgets = state.get(BUDGET_STATE_KEY) if state_writable else None
        if not isinstance(budgets, dict):
            budgets = {}
            if state_writable:
                state[BUDGET_STATE_KEY] = budgets
        agent_budgets = budgets.setdefault(agent_name, {})
        if isinstance(agent_budgets, dict):
            used_now = self._counts.get(key, 0)
            agent_budgets[self.budget_kind] = {
                "limit": self.max_calls_per_agent,
                "used": used_now,
                "remaining": max(0, self.max_calls_per_agent - used_now),
                "status": "exhausted"
                if used_now >= self.max_calls_per_agent
                else "available",
            }

        if key in self._exhausted_agents:
            if self.telemetry is not None:
                self.telemetry.budget_blocked(agent_name)
                self.telemetry.record_blocked(
                    agent_name,
                    tool.name,
                    tool_args,
                    category="budget_exhausted",
                    call_id=getattr(tool_context, "function_call_id", None),
                    budget={
                        "kind": self.budget_kind,
                        "limit": self.max_calls_per_agent,
                        "used": used_now,
                        "remaining": 0,
                        "status": "exhausted",
                    },
                )
            return _finalize_result(agent_name, self.budget_kind)

        url = _normalise_url(tool_args.get("url"))
        if self.reject_empty_urls and "url" in tool_args and not url:
            if self.telemetry is not None:
                self.telemetry.blocked(agent_name)
                self.telemetry.record_blocked(
                    agent_name,
                    tool.name,
                    tool_args,
                    category="invalid_input",
                    call_id=getattr(tool_context, "function_call_id", None),
                )
            return _limit_result(
                "Empty URL rejected for web tool "
                f"'{tool.name}' on agent '{agent_name}'."
            )
        if self.ignore_local_urls and _is_local_url(url):
            _log.debug(
                "Web limit ignored local URL | agent={} tool={} url={}",
                agent_name,
                tool.name,
                url,
            )
            return None

        call_key: tuple[str, str, str] | None = None
        if self.dedupe_identical_calls:
            call_key = (session_id, agent_name, _call_fingerprint(tool, tool_args))
            if call_key in self._seen_calls:
                _log.debug(
                    "Web limit blocked duplicate call | agent={} tool={}",
                    agent_name,
                    tool.name,
                )
                if self.telemetry is not None:
                    self.telemetry.duplicate(agent_name)
                    self.telemetry.record_blocked(
                        agent_name,
                        tool.name,
                        tool_args,
                        category="duplicate_call",
                        call_id=getattr(tool_context, "function_call_id", None),
                    )
                return {
                    "error_code": DUPLICATE_TOOL_CALL,
                    "message": "Identical tool call already made. Use its earlier result or change the query or URL.",
                }

        url_key: tuple[str, str, str] | None = None
        if url:
            url_key = (session_id, agent_name, url)
            if (
                tool_name in self._same_url_exempt_tool_names
                and url_key in self._seen_urls
            ):
                _log.debug(
                    "Web limit ignored same-URL helper call | agent={} tool={} url={}",
                    agent_name,
                    tool.name,
                    url,
                )
                return None
            if self.count_unique_urls and url_key in self._seen_urls:
                _log.debug(
                    "Web limit ignored already-counted URL | agent={} tool={} url={}",
                    agent_name,
                    tool.name,
                    url,
                )
                return None

        used = self._counts.get(key, 0)
        if used >= self.max_calls_per_agent:
            message = (
                "Web search limit exceeded for agent "
                f"'{agent_name}': max {self.max_calls_per_agent} calls per run."
            )
            _log.warning(message)
            self._exhausted_agents.add(key)
            if self.telemetry is not None:
                self.telemetry.exhausted(agent_name, self.budget_kind)
                self.telemetry.record_blocked(
                    agent_name,
                    tool.name,
                    tool_args,
                    category="budget_exhausted",
                    call_id=getattr(tool_context, "function_call_id", None),
                    budget={
                        "kind": self.budget_kind,
                        "limit": self.max_calls_per_agent,
                        "used": used,
                        "remaining": 0,
                        "status": "exhausted",
                    },
                )
            next_step = (
                "Stop discovery searches. Inspect already-found URLs with available "
                "extraction tools, then synthesize from the evidence."
                if self.budget_kind == "search"
                else "Stop extraction and synthesize from the evidence already gathered."
            )
            return _limit_result(
                f"{message} This agent's {self.budget_kind} budget is exhausted. {next_step}",
                error_code=WEB_BUDGET_EXHAUSTED,
            )

        self._counts[key] = used + 1
        if isinstance(agent_budgets, dict):
            remaining = max(0, self.max_calls_per_agent - used - 1)
            budget_state = {
                "limit": self.max_calls_per_agent,
                "used": used + 1,
                "remaining": remaining,
                "status": "exhausted" if remaining == 0 else "available",
            }
            agent_budgets[self.budget_kind] = budget_state
            if self.telemetry is not None:
                self.telemetry.record_budget(
                    agent_name,
                    tool.name,
                    tool_args,
                    {"kind": self.budget_kind, **budget_state},
                    call_id=getattr(tool_context, "function_call_id", None),
                )
        if call_key is not None:
            self._seen_calls.add(call_key)
        if url_key is not None:
            self._seen_urls.add(url_key)
        _log.debug(
            "Web search call allowed | agent={} tool={} used={}/{}",
            agent_name,
            tool.name,
            used + 1,
            self.max_calls_per_agent,
        )
        return None

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> None:
        budgets = callback_context.state.get(BUDGET_STATE_KEY, {})
        agent_name = callback_context._invocation_context.agent.name
        current = budgets.get(agent_name, {}) if isinstance(budgets, dict) else {}
        if not isinstance(current, dict):
            return
        if not (
            isinstance(current.get(self.budget_kind), dict)
            and current[self.budget_kind].get("status") == "exhausted"
        ):
            return
        names = {
            name
            for name, tool in llm_request.tools_dict.items()
            if self._is_web_search_tool(tool)
        }
        if names:
            retained = []
            for group in llm_request.config.tools or []:
                declarations = group.function_declarations
                if declarations is None:
                    retained.append(group)
                    continue
                group.function_declarations = [
                    item for item in declarations if item.name not in names
                ]
                if group.function_declarations:
                    retained.append(group)
            llm_request.config.tools = retained
        llm_request.append_instructions(
            [
                (
                    f"The {self.budget_kind} tool budget is exhausted for this agent. "
                    "Those tools are unavailable now. Continue from collected evidence "
                    "or state which required evidence remains unresolved."
                )
            ]
        )

    def _is_web_search_tool(self, tool: BaseTool) -> bool:
        name = strip_tool_name_prefix(tool.name).lower()
        description = (tool.description or "").lower()
        if self.budget_kind == "search":
            if self._custom_tool_names and name not in self._tool_names:
                return False
            capability = tool_capability(tool.name, description=description)
            if self._custom_tool_names and name != "search":
                return True
            if capability != ToolCapability.DISCOVERY:
                return False
            if normalize_tool_name(tool.name) == "search":
                description = (tool.description or "").casefold()
                return any(
                    hint in description
                    for hint in _WEB_SEARCH_DESCRIPTION_HINTS
                )
            return True
        if self.budget_kind == "scraping":
            return is_inspection_tool(tool.name)
        if self.budget_kind in {"browser", "browser_agent"}:
            return tool_capability(tool.name) == ToolCapability.BROWSER_NAVIGATION
        if self._custom_tool_names:
            return name in self._tool_names
        return False


class WebSearchLimitExceeded(RuntimeError):
    """Legacy exception type retained for import compatibility; no longer raised.

    Budget exhaustion is now returned as a synthetic tool result carrying the
    ``WEB_BUDGET_EXHAUSTED`` error code. Raising from ADK callbacks can abort a
    parallel tool-call batch and orphan its correlated responses.
    """


def _limit_result(message: str, *, error_code: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"isError": True, "error": message}
    if error_code is not None:
        result["error_code"] = error_code
    return result


def _finalize_result(agent_name: str, budget_kind: str) -> dict[str, Any]:
    next_step = (
        "Inspect already-found URLs with available extraction tools, then synthesize."
        if budget_kind == "search"
        else "Synthesize from the evidence already gathered."
    )
    return _limit_result(
        f"{budget_kind.capitalize()} tools are disabled for agent '{agent_name}' "
        f"because its budget is exhausted. {next_step}",
        error_code=WEB_BUDGET_EXHAUSTED,
    )


def _normalise_url(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    value = value.strip()
    if not value:
        return ""
    if value.startswith("file://"):
        return value
    try:
        split = urlsplit(value)
    except ValueError:
        return value
    if split.scheme not in {"http", "https"}:
        return value
    split = split._replace(query="&".join(sorted(split.query.split("&"))))
    return urldefrag(urlunsplit(split)).url


def _is_local_url(url: str) -> bool:
    return url.startswith("file://")


def _call_fingerprint(tool: BaseTool, tool_args: dict[str, Any]) -> str:
    return f"{tool.name.lower()}:{_freeze(tool_args)!r}"


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((key, _freeze(item)) for key, item in value.items()))
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted(_freeze(item) for item in value))
    return value
