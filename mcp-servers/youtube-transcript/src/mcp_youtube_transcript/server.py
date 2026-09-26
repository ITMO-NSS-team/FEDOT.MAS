from __future__ import annotations

import asyncio
import json
import os
from urllib.parse import parse_qs, urlparse

import mcp.types as mt
from fastmcp.client.transports import StdioTransport
from fastmcp.server import create_proxy
from fastmcp.server.middleware import CallNext, Middleware, MiddlewareContext
from fastmcp.tools import ToolResult
from youtube_transcript_api import (
    AgeRestricted,
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
    YouTubeTranscriptApi,
)

TRANSCRIPT_TOOLS = frozenset(
    {"get_transcript", "get_timed_transcript", "get_available_languages"}
)

VIDEO_URL_TOOLS = frozenset(
    {
        "get_video_info",
        "get_transcript",
        "get_timed_transcript",
        "get_available_languages",
    }
)
def _error(code: str, message: str) -> ToolResult:
    payload = {"error_code": code, "message": message}
    return ToolResult(
        content=[mt.TextContent(type="text", text=json.dumps(payload))],
        structured_content=payload,
        is_error=True,
    )


def _video_id(url: str) -> str | None:
    parsed = urlparse(url)
    if parsed.hostname == "youtu.be":
        return parsed.path.strip("/") or None
    if parsed.hostname not in {"youtube.com", "www.youtube.com", "m.youtube.com"}:
        return None
    if parsed.path.startswith(("/shorts/", "/embed/", "/live/")):
        return parsed.path.split("/")[2] or None
    return (parse_qs(parsed.query).get("v") or [None])[0]


def _diagnose_transcript_failure(url: str, lang: str) -> ToolResult | None:
    video_id = _video_id(url)
    if video_id is None:
        return None
    try:
        languages = ["en"] if lang == "en" else [lang, "en"]
        YouTubeTranscriptApi().list(video_id).find_transcript(languages).fetch()
    except TranscriptsDisabled:
        return _error("TRANSCRIPTS_DISABLED", "Subtitles are disabled for this video.")
    except NoTranscriptFound:
        return _error(
            "NO_TRANSCRIPT_FOUND",
            "No transcript is available in the requested language.",
        )
    except VideoUnavailable:
        return _error("VIDEO_UNAVAILABLE", "This video is unavailable.")
    except AgeRestricted:
        return _error("AGE_RESTRICTED", "This video requires age verification.")
    except Exception:  # noqa: BLE001 - diagnostic failure must preserve upstream error
        return None
    return None


class TranscriptErrors(Middleware):
    async def on_call_tool(
        self,
        context: MiddlewareContext[mt.CallToolRequestParams],
        call_next: CallNext[mt.CallToolRequestParams, ToolResult],
    ) -> ToolResult:
        if context.message.name not in TRANSCRIPT_TOOLS:
            return await call_next(context)

        arguments = context.message.arguments or {}
        if context.message.name in VIDEO_URL_TOOLS:
            url = arguments.get("url")
            if not isinstance(url, str) or _video_id(url) is None:
                return _error(
                    "INVALID_VIDEO_URL",
                    "Expected a direct YouTube video URL, not a homepage, channel, or search page.",
                )
        if "next_cursor" in arguments and arguments["next_cursor"] is not None:
            cursor = arguments["next_cursor"]
            if (
                isinstance(cursor, bool)
                or not str(cursor).isascii()
                or not str(cursor).isdecimal()
            ):
                return _error(
                    "INVALID_CURSOR", "next_cursor must be a non-negative integer."
                )
            arguments["next_cursor"] = str(cursor)
            context.message.arguments = arguments

        result = await call_next(context)
        if not result.is_error:
            return result

        url = arguments.get("url")
        if not isinstance(url, str):
            return result
        lang = arguments.get("lang", "en")
        if not isinstance(lang, str):
            return result
        diagnosed = await asyncio.to_thread(_diagnose_transcript_failure, url, lang)
        return diagnosed or result


transport = StdioTransport(
    command="uvx",
    args=[
        "--from",
        "git+https://github.com/jkawamoto/mcp-youtube-transcript",
        "mcp-youtube-transcript",
    ],
    env=dict(os.environ),
)

mcp = create_proxy(transport, name="youtube-transcript")
mcp.add_middleware(TranscriptErrors())


def main():
    mcp.run(show_banner=False)
