"""Shared capability classification for MCP tools used by the runtime."""

from __future__ import annotations

from enum import StrEnum


class ToolCapability(StrEnum):
    DISCOVERY = "discovery"
    URL_INSPECTION = "url_inspection"
    DOCUMENT_INSPECTION = "document_inspection"
    MEDIA_INSPECTION = "media_inspection"
    BROWSER_NAVIGATION = "browser_navigation"
    COMPUTATION = "computation"
    DIAGNOSTIC = "diagnostic/control"
    OTHER = "other"


_DISCOVERY = frozenset(
    {
        "web_search",
        "websearch",
        "google_search",
        "searxng_search",
        "tavily_search",
    }
)
_URL_INSPECTION = frozenset(
    {"goto", "markdown", "extract", "links", "eval", "evaluate", "screenshot"}
)
_DOCUMENT_INSPECTION = frozenset(
    {
        "read_document",
        "find_document",
        "download",
        "download_file",
        "read_file",
        "extract_zip",
        "list_zip_contents",
    }
)
_MEDIA_INSPECTION = frozenset(
    {
        "get_video_info",
        "get_transcript",
        "get_timed_transcript",
        "transcribe_audio",
        "analyze_audio",
        "analyze_video",
        "analyze_image",
    }
)
_BROWSER = frozenset({"complete_browser_task"})
_COMPUTATION = frozenset({"solve_with_code", "run_python", "execute_code"})
_DIAGNOSTIC = frozenset(
    {"telemetry", "health", "health_check", "get_metrics", "diagnostics"}
)
_DISCOVERY_SERVERS = frozenset({"websearch_tavily", "websearch_searxng"})
_INSPECTION_SERVERS = frozenset(
    {"web_scraping", "document", "download", "youtube_transcript", "browser_agent"}
)


def normalize_tool_name(name: str) -> str:
    """Strip ADK server namespaces/prefixes to classify a bare MCP tool name."""
    normalized = name.rsplit("__", 1)[-1].lower().replace("-", "_").replace(".", "_")
    # MCP server prefixes are conventionally prepended with an underscore. Strip
    # a prefix only when the suffix is a known capability-bearing tool name.
    for known in (
        *_DISCOVERY,
        *_URL_INSPECTION,
        *_DOCUMENT_INSPECTION,
        *_MEDIA_INSPECTION,
        *_BROWSER,
        *_COMPUTATION,
        *_DIAGNOSTIC,
        *_DISCOVERY_SERVERS,
        *_INSPECTION_SERVERS,
    ):
        if normalized == known or normalized.endswith(f"_{known}"):
            return known
    return normalized


def tool_capability(name: str, *, description: str = "", server: str = "") -> ToolCapability:
    """Return the runtime capability for a bare, prefixed, or namespaced tool."""
    raw = name.lower().replace("-", "_").replace(".", "_")
    server_name = server.lower().replace("-", "_").replace(".", "_")
    normalized = normalize_tool_name(name)
    if normalized in _DIAGNOSTIC:
        return ToolCapability.DIAGNOSTIC
    if normalized in _DISCOVERY_SERVERS or (
        normalized in {"search", "web_search"}
        and any(s in raw or s in server_name for s in _DISCOVERY_SERVERS)
    ):
        return ToolCapability.DISCOVERY
    if normalized == "search":
        hints = ("web search", "search the web", "search the internet", "internet search", "search engine", "search broadly", "search for independent", "search sources", "tavily", "searx")
        return ToolCapability.DISCOVERY if any(hint in description.casefold() for hint in hints) else ToolCapability.OTHER
    if normalized in {"web_search", "websearch", "google_search", "searxng_search", "tavily_search"} or normalized.endswith(("_web_search", "_google_search", "_searxng_search", "_tavily_search")):
        return ToolCapability.DISCOVERY
    if normalized == "web_scraping" or normalized in _URL_INSPECTION:
        return ToolCapability.URL_INSPECTION
    if normalized in _DOCUMENT_INSPECTION or normalized in {"document", "download"}:
        return ToolCapability.DOCUMENT_INSPECTION
    if normalized in _MEDIA_INSPECTION or normalized == "youtube_transcript":
        return ToolCapability.MEDIA_INSPECTION
    if normalized in _BROWSER or normalized == "browser_agent":
        return ToolCapability.BROWSER_NAVIGATION
    if normalized in _COMPUTATION:
        return ToolCapability.COMPUTATION
    if (
        normalized in _DIAGNOSTIC
        or normalized.endswith(("_telemetry", "_diagnostics"))
    ):
        return ToolCapability.DIAGNOSTIC
    return ToolCapability.OTHER


def is_solving_tool(name: str) -> bool:
    """Whether a tool belongs on an ordinary agent's model-visible surface."""
    return tool_capability(name) != ToolCapability.DIAGNOSTIC


def is_inspection_tool(name: str) -> bool:
    return tool_capability(name) in {
        ToolCapability.URL_INSPECTION,
        ToolCapability.DOCUMENT_INSPECTION,
        ToolCapability.MEDIA_INSPECTION,
        ToolCapability.BROWSER_NAVIGATION,
    }
