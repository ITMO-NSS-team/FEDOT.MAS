from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

from google.adk.plugins import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext

from fedotmas.common.logging import get_logger
from fedotmas.mcp import strip_tool_name_prefix

_log = get_logger("fedotmas.plugins.browser_fallback_policy")

BROWSER_TOOL_NAMES = frozenset(
    {"goto", "markdown", "links", "extract", "eval", "evaluate", "screenshot"}
)

DOCUMENT_EXTENSIONS = frozenset(
    {
        ".pdf",
        ".doc",
        ".docx",
        ".ppt",
        ".pptx",
        ".xls",
        ".xlsx",
        ".csv",
        ".json",
        ".xml",
        ".html",
        ".htm",
        ".txt",
    }
)
# Remote URLs ending in these are better downloaded + parsed than browsed.
# HTML pages are deliberately excluded: normal web articles should be navigated
# with the browser, not forced through download + document.read_document (which
# adds latency, tool errors, and retries that drive task timeouts).
REMOTE_DOCUMENT_EXTENSIONS = frozenset(
    {".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx", ".csv", ".json", ".xml"}
)
IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff", ".svg", ".ico"}
)
AUDIO_EXTENSIONS = frozenset(
    {".mp3", ".wav", ".flac", ".oga", ".ogg", ".aiff", ".aac", ".m4a", ".wma", ".opus"}
)
VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".webm", ".mpeg", ".mpg", ".avi", ".mkv"})
MEDIA_EXTENSIONS = IMAGE_EXTENSIONS | AUDIO_EXTENSIONS | VIDEO_EXTENSIONS

NAVIGATION_FALLBACK_ERRORS = (
    "SslConnectError",
    "OperationTimedout",
    "UnsupportedProtocol",
    "navigate failed",
)


class BrowserFallbackPolicyPlugin(BasePlugin):
    """Prevent browser misuse and convert navigation failures into fallback hints."""

    def __init__(
        self,
        *,
        browser_tool_names: set[str] | None = None,
        name: str = "fedotmas_browser_fallback_policy",
    ) -> None:
        super().__init__(name=name)
        self._browser_tool_names = {
            item.lower() for item in (browser_tool_names or BROWSER_TOOL_NAMES)
        }

    async def before_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
    ) -> Optional[dict]:
        if not self._is_browser_tool(tool):
            return None

        target = _target_from_args(tool_args)
        if not target:
            return None

        route = _route_for_target(target)
        if route is None:
            return None

        agent_name = tool_context._invocation_context.agent.name  # noqa: E501  # ty: ignore[unresolved-attribute]
        _log.warning(
            "Browser call blocked by fallback policy | agent={} tool={} target={} fallback={}",
            agent_name,
            tool.name,
            target,
            route["recommended_tool"],
        )
        return {
            "browser_blocked": True,
            "reason": route["reason"],
            "target": target,
            "recommended_tool": route["recommended_tool"],
            "recommended_next_action": route["recommended_next_action"],
        }

    async def after_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        result: dict,
    ) -> Optional[dict]:
        if not self._is_browser_tool(tool):
            return None

        result_text = _stringify_result(result)
        if not any(error in result_text for error in NAVIGATION_FALLBACK_ERRORS):
            return None

        target = _target_from_args(tool_args)
        fallback = _fallback_for_navigation_failure(target, result_text)
        agent_name = tool_context._invocation_context.agent.name  # noqa: E501  # ty: ignore[unresolved-attribute]
        _log.warning(
            "Browser navigation fallback suggested | agent={} tool={} target={} fallback={}",
            agent_name,
            tool.name,
            target or "",
            fallback["recommended_tool"],
        )
        return {
            "browser_navigation_failed": True,
            "target": target,
            "error": _compact_error(result_text),
            "recommended_tool": fallback["recommended_tool"],
            "recommended_next_action": fallback["recommended_next_action"],
        }

    def _is_browser_tool(self, tool: BaseTool) -> bool:
        # Through the prefix: BROWSER_TOOL_NAMES holds the bare names, and a
        # server declaring tool_name_prefix renames all of its tools.
        return strip_tool_name_prefix(tool.name).lower() in self._browser_tool_names


def _target_from_args(tool_args: dict[str, Any]) -> str:
    for key in ("url", "uri", "href", "target", "file_path", "path"):
        value = tool_args.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _route_for_target(target: str) -> dict[str, str] | None:
    if _is_local_target(target):
        ext = _extension_for_target(target)
        if ext in AUDIO_EXTENSIONS:
            return _route(
                "local audio files should be transcribed by the media tool, not opened in browser",
                "media.transcribe_audio",
                "Call transcribe_audio with this local file path, then answer from the transcript.",
            )
        if ext in IMAGE_EXTENSIONS:
            return _route(
                "local image files should be analyzed by the media tool, not opened in browser",
                "media.analyze_image",
                "Call analyze_image with this local file path and a task-specific prompt.",
            )
        if ext in VIDEO_EXTENSIONS:
            return _route(
                "local video files should be analyzed by the media tool, not opened in browser",
                "media.analyze_video",
                "Call analyze_video with this local file path and a task-specific prompt.",
            )
        if ext in DOCUMENT_EXTENSIONS:
            return _route(
                "local document/data files should be read by document or sandbox tools, not opened in browser",
                "document.read_document",
                "Call read_document for text extraction; use sandbox for structured computation when needed.",
            )
        return _route(
            "local file URLs are not browser navigation targets",
            "document/media/sandbox",
            "Choose the tool by file extension; inspect the local file directly without browser.goto.",
        )

    if _is_archive_org(target):
        return _route(
            "archive.org pages are heavy browser targets",
            "archive metadata/download API",
            "Use archive.org metadata/download URLs to list files and download the needed document directly.",
        )

    ext = _extension_for_target(target)
    if ext in MEDIA_EXTENSIONS:
        return _route(
            "direct media URLs are better handled by download plus media tools",
            "download + media",
            "Download the URL directly, then call the matching media tool on the downloaded file.",
        )
    if ext in REMOTE_DOCUMENT_EXTENSIONS:
        return _route(
            "direct document/data URLs are better handled by download/document tools",
            "download + document.read_document",
            "Download the URL directly, then parse it with document.read_document or sandbox.",
        )
    return None


def _fallback_for_navigation_failure(target: str, error_text: str) -> dict[str, str]:
    route = _route_for_target(target) if target else None
    if route is not None:
        return route

    if target and _is_archive_org(target):
        return _route(
            "browser navigation failed on archive.org",
            "archive metadata/download API",
            "Use archive.org metadata/download URLs instead of loading the page in browser.",
        )
    if "SslConnectError" in error_text:
        return _route(
            "browser SSL navigation failed",
            "download",
            "Try direct HTTP download/fetch with redirects; if it is media, pass the downloaded file to media tools.",
        )
    if "OperationTimedout" in error_text:
        return _route(
            "browser navigation timed out",
            "download/API",
            "Avoid retrying the same browser navigation; use direct download, site API, or a narrower URL.",
        )
    if "UnsupportedProtocol" in error_text:
        return _route(
            "browser does not support this URL protocol",
            "document/media/sandbox",
            "Use a local-file capable tool selected by file extension.",
        )
    return _route(
        "browser navigation failed",
        "download/API/specialized tool",
        "Use a non-browser fallback rather than retrying the same navigation.",
    )


def _route(reason: str, recommended_tool: str, next_action: str) -> dict[str, str]:
    return {
        "reason": reason,
        "recommended_tool": recommended_tool,
        "recommended_next_action": next_action,
    }


def _is_local_target(target: str) -> bool:
    parsed = urlparse(target)
    return parsed.scheme == "file" or (not parsed.scheme and Path(target).is_absolute())


def _is_archive_org(target: str) -> bool:
    return urlparse(target).netloc.lower().endswith("archive.org")


def _extension_for_target(target: str) -> str:
    parsed = urlparse(target)
    path = parsed.path if parsed.scheme else target
    return Path(path).suffix.lower()


def _stringify_result(result: Any) -> str:
    if isinstance(result, dict):
        return " ".join(
            f"{key}={_stringify_result(value)}" for key, value in result.items()
        )
    if isinstance(result, list):
        return " ".join(_stringify_result(item) for item in result)
    return str(result)


def _compact_error(error_text: str) -> str:
    error_text = " ".join(error_text.split())
    return error_text[:500]
