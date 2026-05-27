from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from fedotmas.plugins import BrowserFallbackPolicyPlugin


def _tool(name: str = "goto") -> MagicMock:
    tool = MagicMock()
    tool.name = name
    return tool


def _tool_context(agent_name: str = "browser_agent") -> MagicMock:
    context = MagicMock()
    context._invocation_context = SimpleNamespace(
        agent=SimpleNamespace(name=agent_name)
    )
    return context


class TestBrowserFallbackPolicyPlugin:
    @pytest.mark.asyncio
    async def test_blocks_local_audio_browser_navigation(self):
        plugin = BrowserFallbackPolicyPlugin()

        result = await plugin.before_tool_callback(
            tool=_tool("goto"),
            tool_args={"url": "file:///tmp/input.mp3"},
            tool_context=_tool_context(),
        )

        assert result is not None
        assert result["browser_blocked"] is True
        assert result["recommended_tool"] == "media.transcribe_audio"
        assert "transcribe_audio" in result["recommended_next_action"]

    @pytest.mark.asyncio
    async def test_blocks_local_document_browser_navigation(self):
        plugin = BrowserFallbackPolicyPlugin()

        result = await plugin.before_tool_callback(
            tool=_tool("goto"),
            tool_args={"url": "/tmp/report.pdf"},
            tool_context=_tool_context(),
        )

        assert result is not None
        assert result["browser_blocked"] is True
        assert result["recommended_tool"] == "document.read_document"

    @pytest.mark.asyncio
    async def test_suggests_download_for_ssl_failure(self):
        plugin = BrowserFallbackPolicyPlugin()

        result = await plugin.after_tool_callback(
            tool=_tool("goto"),
            tool_args={"url": "https://example.com/image.jpg"},
            tool_context=_tool_context(),
            result={"isError": True, "error": "navigate failed err=SslConnectError"},
        )

        assert result is not None
        assert result["browser_navigation_failed"] is True
        assert result["recommended_tool"] == "download + media"

    @pytest.mark.asyncio
    async def test_suggests_archive_api_for_archive_timeout(self):
        plugin = BrowserFallbackPolicyPlugin()

        result = await plugin.after_tool_callback(
            tool=_tool("goto"),
            tool_args={
                "url": "https://archive.org/details/lego-building-instructions-4852"
            },
            tool_context=_tool_context(),
            result={"isError": True, "error": "navigate failed err=OperationTimedout"},
        )

        assert result is not None
        assert result["browser_navigation_failed"] is True
        assert result["recommended_tool"] == "archive metadata/download API"

    @pytest.mark.asyncio
    async def test_non_browser_tool_ignored(self):
        plugin = BrowserFallbackPolicyPlugin()

        result = await plugin.before_tool_callback(
            tool=_tool("read_document"),
            tool_args={"file_path": "/tmp/report.pdf"},
            tool_context=_tool_context(),
        )

        assert result is None
