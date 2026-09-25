from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fedotmas.plugins import ToolResultTruncationPlugin
from fedotmas.plugins._tool_result_truncation import _truncate_total
from google.adk.models.llm_request import LlmRequest
from google.genai import types


def _tool(name: str = "markdown") -> MagicMock:
    tool = MagicMock()
    tool.name = name
    return tool


def _tool_context(*, agent_name: str = "researcher"):
    ctx = MagicMock()
    ctx._invocation_context.agent.name = agent_name
    return ctx


class TestToolResultTruncationPlugin:
    @pytest.mark.parametrize("limit", [1, 2])
    def test_structural_overhead_cannot_make_total_truncation_stall(self, limit):
        value, changed = _truncate_total({"payload": [1]}, limit)

        assert changed is True
        assert len(json.dumps(value, ensure_ascii=False, default=str)) <= limit

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
    async def test_old_results_are_compacted_and_newest_payload_remains_active(self):
        plugin = ToolResultTruncationPlugin(
            max_string_chars=5000,
            max_total_chars=3000,
            aggregate_tool_names={"*"},
            max_agent_total_chars=500,
        )
        contents = []
        for number in range(12):
            contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_function_response(
                            name="read_document",
                            response={
                                "url": f"https://example.org/{number}",
                                "source_urls": [
                                    f"https://example.org/{number}/source"
                                ],
                                "title": f"Source {number}",
                                "content": "old evidence " + "x" * 2000,
                            },
                        )
                    ],
                )
            )
        contents.append(
            types.Content(
                role="user",
                parts=[
                    types.Part.from_function_response(
                        name="search",
                        response={
                            "results": [
                                {
                                    "url": "https://example.org/latest",
                                    "title": "Latest source",
                                    "snippet": "TARGETED EVIDENCE",
                                    "content": "TARGETED EVIDENCE " + "y" * 800,
                                    "value": "latest payload",
                                }
                            ]
                        },
                    )
                ],
            )
        )
        request = LlmRequest(contents=contents)

        await plugin.before_model_callback(
            callback_context=SimpleNamespace(), llm_request=request
        )

        responses = [
            part.function_response.response
            for content in request.contents
            for part in content.parts or []
            if part.function_response is not None
        ]
        active_chars = sum(len(json.dumps(response)) for response in responses)
        assert active_chars <= 500
        assert responses[-1]["results"][0]["snippet"] == "TARGETED EVIDENCE"
        assert responses[-1]["results"][0]["content"].startswith(
            "TARGETED EVIDENCE"
        )
        assert responses[-1]["results"][0]["value"] == "latest payload"
        compacted = [
            response for response in responses[:-1] if response
        ]
        assert compacted
        assert all("url" in response for response in compacted)
        assert all("source_urls" in response for response in compacted)
        assert all("content" not in response for response in compacted)

    @pytest.mark.asyncio
    async def test_rolling_compaction_preserves_controller_research_state(self):
        plugin = ToolResultTruncationPlugin(
            max_string_chars=500,
            max_total_chars=1000,
            aggregate_tool_names={"*"},
            max_agent_total_chars=500,
        )
        research_state = {
            "research_id": "workstream-1",
            "telemetry": {
                "pending_recommendation": {
                    "id": 4,
                    "action": "continue_search",
                    "searches_before": 3,
                }
            },
        }
        latest = await plugin.after_tool_callback(
            tool=_tool("get_next_action"),
            tool_args={},
            tool_context=_tool_context(),
            result={
                "action": "continue_search",
                "guidance": "x" * 800,
                "research_state": research_state,
                "results": [{"content": "old evidence " * 80}],
            },
        )
        assert latest is not None
        assert latest["research_state"] == research_state

        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_function_response(
                        name="search",
                        response={"url": f"https://example.org/{i}", "content": "z" * 900},
                    )
                ],
            )
            for i in range(5)
        ]
        contents.append(
            types.Content(
                role="user",
                parts=[
                    types.Part.from_function_response(
                        name="get_next_action", response=latest
                    )
                ],
            )
        )
        request = LlmRequest(contents=contents)

        await plugin.before_model_callback(
            callback_context=SimpleNamespace(), llm_request=request
        )

        responses = [
            part.function_response.response
            for content in request.contents
            for part in content.parts or []
            if part.function_response is not None
        ]
        active_chars = sum(len(json.dumps(response)) for response in responses)
        pending = responses[-1]["research_state"]["telemetry"][
            "pending_recommendation"
        ]
        assert active_chars <= 500
        assert pending == {
            "id": 4,
            "action": "continue_search",
            "searches_before": 3,
        }

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
