from __future__ import annotations

from typing import Any, Optional
from urllib.parse import urldefrag, urlsplit, urlunsplit

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
        count_unique_urls: bool = False,
        same_url_exempt_tool_names: set[str] | None = None,
        ignore_local_urls: bool = True,
        reject_empty_urls: bool = False,
        dedupe_identical_calls: bool = True,
        hard_fail: bool = False,
        name: str = "fedotmas_web_search_limit",
    ) -> None:
        if max_calls_per_agent < 1:
            raise ValueError("max_calls_per_agent must be >= 1")
        super().__init__(name=name)
        self.max_calls_per_agent = max_calls_per_agent
        self.hard_fail = hard_fail
        # When True, every matched web/search tool call is blocked with a soft
        # "answer now" result (never raises). Set during the post-budget
        # finalization turn so the agent stops exploring and commits an answer.
        self.finalizing = False
        self.count_unique_urls = count_unique_urls
        self.ignore_local_urls = ignore_local_urls
        self.reject_empty_urls = reject_empty_urls
        self.dedupe_identical_calls = dedupe_identical_calls
        self._tool_names = {
            name.lower() for name in (tool_names or DEFAULT_WEB_SEARCH_TOOL_NAMES)
        }
        self._same_url_exempt_tool_names = {
            name.lower() for name in (same_url_exempt_tool_names or set())
        }
        self._counts: dict[tuple[str, str], int] = {}
        self._seen_calls: set[tuple[str, str, str]] = set()
        self._seen_urls: set[tuple[str, str, str]] = set()

    async def before_run_callback(
        self, *, invocation_context: InvocationContext
    ) -> None:
        session_id = invocation_context.session.id
        self._counts = {
            key: count for key, count in self._counts.items() if key[0] != session_id
        }
        self._seen_calls = {key for key in self._seen_calls if key[0] != session_id}
        self._seen_urls = {key for key in self._seen_urls if key[0] != session_id}
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

        if self.finalizing:
            return _limit_result(
                "Search/exploration budget exhausted and tools are now disabled. "
                "Do not call web, browser, or search tools. Provide your best final "
                "answer from the evidence already gathered."
            )

        session_id = tool_context._invocation_context.session.id
        agent_name = tool_context._invocation_context.agent.name
        tool_name = tool.name.lower()
        key = (session_id, agent_name)

        url = _normalise_url(tool_args.get("url"))
        if self.reject_empty_urls and "url" in tool_args and not url:
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
                    "Web limit ignored duplicate call | agent={} tool={}",
                    agent_name,
                    tool.name,
                )
                return None

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
            if self.count_unique_urls:
                if url_key in self._seen_urls:
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
            if self.hard_fail:
                raise WebSearchLimitExceeded(message)
            return _limit_result(message)

        self._counts[key] = used + 1
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

    def _is_web_search_tool(self, tool: BaseTool) -> bool:
        name = tool.name.lower()
        if name in self._tool_names:
            description = (tool.description or "").lower()
            if name == "search":
                return any(hint in description for hint in WEB_SEARCH_HINTS)
            return True
        return False


class WebSearchLimitExceeded(RuntimeError):
    """Raised when a hard web-search/tool budget is exhausted."""


def _limit_result(message: str) -> dict[str, Any]:
    return {"isError": True, "error": message}


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
