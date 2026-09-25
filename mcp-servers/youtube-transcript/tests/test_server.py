from __future__ import annotations

import asyncio
import json

import mcp.types as mt
import pytest
from fastmcp.server.middleware import MiddlewareContext
from fastmcp.tools import ToolResult
from mcp_youtube_transcript import server
from youtube_transcript_api import TranscriptsDisabled


def _run(name, arguments, response):
    context = MiddlewareContext(
        message=mt.CallToolRequestParams(name=name, arguments=arguments)
    )
    calls = []

    async def call_next(current):
        calls.append(current.message.arguments)
        return response

    result = asyncio.run(server.TranscriptErrors().on_call_tool(context, call_next))
    return result, calls


def _payload(result):
    return json.loads(result.content[0].text)


def test_disabled_subtitles_are_returned_as_structured_error(monkeypatch):
    class DisabledApi:
        def list(self, video_id):
            assert video_id == "abc123"
            raise TranscriptsDisabled(video_id)

    monkeypatch.setattr(server, "YouTubeTranscriptApi", DisabledApi)
    generic_error = ToolResult(
        content=[
            mt.TextContent(type="text", text="Error executing tool get_transcript")
        ],
        is_error=True,
    )

    result, calls = _run(
        "get_transcript",
        {"url": "https://www.youtube.com/watch?v=abc123"},
        generic_error,
    )

    assert len(calls) == 1
    assert result.is_error
    assert result.structured_content == {
        "error_code": "TRANSCRIPTS_DISABLED",
        "message": "Subtitles are disabled for this video.",
    }
    assert _payload(result) == result.structured_content


@pytest.mark.parametrize("name", ["get_transcript", "get_timed_transcript"])
@pytest.mark.parametrize("cursor", ["abc", "1.5", "-1", "", True])
def test_invalid_cursor_returns_structured_error_without_calling_upstream(name, cursor):
    response = ToolResult(content="unused")

    result, calls = _run(
        name, {"url": "https://youtu.be/abc123", "next_cursor": cursor}, response
    )

    assert calls == []
    assert result.is_error
    assert result.structured_content == {
        "error_code": "INVALID_CURSOR",
        "message": "next_cursor must be a non-negative integer.",
    }
    assert _payload(result) == result.structured_content


def test_integer_cursor_reaches_upstream_as_string():
    response = ToolResult(content="transcript")

    result, calls = _run(
        "get_transcript",
        {"url": "https://youtu.be/abc123", "next_cursor": 12},
        response,
    )

    assert result is response
    assert calls == [{"url": "https://youtu.be/abc123", "next_cursor": "12"}]
