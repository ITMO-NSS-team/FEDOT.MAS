from __future__ import annotations

import json
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
    async def test_aggregate_limit_is_opt_in_for_other_callers(self):
        plugin = ToolResultTruncationPlugin(max_string_chars=1000)
        payload = {"results": [{"snippet": "x" * 500} for _ in range(500)]}

        result = await plugin.after_tool_callback(
            tool=_tool("unrelated_tool"),
            tool_args={},
            tool_context=_tool_context(),
            result=payload,
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_aggregate_nested_search_payload_is_bounded(self):
        plugin = ToolResultTruncationPlugin(
            max_string_chars=1000,
            max_total_chars=1000,
            aggregate_tool_names={"search"},
        )
        payload = {
            "results": [
                {"title": "x" * 80, "url": f"https://example.com/{n}"}
                for n in range(50)
            ]
        }
        result = await plugin.after_tool_callback(
            tool=_tool("search"),
            tool_args={},
            tool_context=_tool_context(),
            result=payload,
        )
        assert result is not None
        assert len(json.dumps(result, ensure_ascii=False)) <= 1000
        assert result["truncated"] is True
        assert len(result["results"]) < 50
        assert (
            await plugin.after_tool_callback(
                tool=_tool("markdown"),
                tool_args={},
                tool_context=_tool_context(),
                result=payload,
            )
            is None
        )

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
    async def test_exhausted_agent_budget_keeps_nested_source_metadata(self):
        plugin = ToolResultTruncationPlugin(
            max_string_chars=1000,
            max_total_chars=1000,
            aggregate_tool_names={"*"},
            max_agent_total_chars=500,
        )
        context = _tool_context()
        context._invocation_context.session.id = "session-1"

        await plugin.after_tool_callback(
            tool=_tool("markdown"),
            tool_args={},
            tool_context=context,
            result={"url": "https://example.org/first", "content": "x" * 30000},
        )
        result = await plugin.after_tool_callback(
            tool=_tool("search"),
            tool_args={},
            tool_context=context,
            result={
                "structuredContent": {
                    "results": [
                        {
                            "url": "https://example.org/second",
                            "title": "Source title",
                            "snippet": "Decisive excerpt",
                            "content": "y" * 30000,
                        }
                    ],
                    "total_results": 1,
                }
            },
        )

        assert result is not None
        assert result["structuredContent"]["results"][0]["url"] == (
            "https://example.org/second"
        )
        assert result["structuredContent"]["results"][0]["snippet"] == (
            "Decisive excerpt"
        )

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
