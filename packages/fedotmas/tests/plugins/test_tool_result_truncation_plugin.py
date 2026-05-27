from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from fedotmas.plugins import ToolResultTruncationPlugin


def _tool(name: str = "markdown") -> MagicMock:
    tool = MagicMock()
    tool.name = name
    return tool


def _tool_context(*, agent_name: str = "researcher"):
    ctx = MagicMock()
    ctx._invocation_context.agent.name = agent_name
    return ctx


class TestToolResultTruncationPlugin:
    @pytest.mark.asyncio
    async def test_ignores_small_result(self):
        plugin = ToolResultTruncationPlugin(max_string_chars=10)

        result = await plugin.after_tool_callback(
            tool=_tool(),
            tool_args={},
            tool_context=_tool_context(),
            result={"content": "small"},
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_truncates_nested_strings(self):
        plugin = ToolResultTruncationPlugin(max_string_chars=5)

        result = await plugin.after_tool_callback(
            tool=_tool(),
            tool_args={},
            tool_context=_tool_context(),
            result={"content": [{"type": "text", "text": "0123456789"}]},
        )

        assert result is not None
        text = result["content"][0]["text"]
        assert text.startswith("01234")
        assert "truncated to 5/10 chars" in text
        assert result["truncated"] is True
        assert result["complete"] is False
        assert result["max_chars"] == 5
        assert "targeted find" in result["recommended_next_action"]
