from __future__ import annotations

import importlib.util
import json
from pathlib import Path
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

    def test_impossibly_small_state_budget_fails_explicitly(self):
        state = {
            "version": 1,
            "research_id": "r",
            "goal": "g",
            "unresolved_questions": [f"{index}-" + "x" * 230 for index in range(30)],
            "search_count": 4,
        }
        with pytest.raises(ValueError, match="too small for valid research_state"):
            _truncate_total({"research_state": state}, 500)

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
    async def test_oversized_research_state_is_schema_compacted_and_stays_actionable(self):
        plugin = ToolResultTruncationPlugin(
            max_string_chars=6000,
            max_total_chars=1000,
            aggregate_tool_names={"*"},
            max_agent_total_chars=1200,
        )
        state = {
            "version": 1,
            "research_id": "workstream-1",
            "goal": "Find a date from an external source",
            "unresolved_questions": ["Which document contains the date?"],
            "required_fields": ["date"],
            "filled_fields": [],
            "search_count": 7,
            "failed_attempt_count": 2,
            "failed_strategy_counts": {"search": 2},
            "remaining_budget": 5,
            "confidence": 0.4,
            "findings": [f"old finding {index} " + "x" * 220 for index in range(30)],
            "evidence": [f"old evidence {index} " + "y" * 220 for index in range(30)],
            "evidence_urls": [f"https://source.example/{index}" for index in range(30)],
            "independent_sources": [f"source-{index}" for index in range(30)],
            "search_intents": [{"terms": ["historic query"], "entities": []}] * 40,
            "failed_approaches": ["old failed approach"] * 30,
            "telemetry": {
                "controller_calls": 9,
                "recommendation_count": 4,
                "followed_recommendations": 2,
                "pending_recommendation": {
                    "id": 4,
                    "action": "continue_search",
                    "searches_before": 7,
                },
                "recommendations": [{"reason": "old"}] * 40,
                "intervention_outcomes": [{"followed": False}] * 40,
            },
        }
        oversized = {"research_state": state, "old_evidence": "z" * 2500}
        compacted, changed = _truncate_total(oversized, 1000)
        serialized = json.dumps(compacted, ensure_ascii=False)
        assert changed is True
        assert len(serialized) <= 1000
        retained = compacted["research_state"]
        assert retained["goal"] == state["goal"]
        assert retained["research_id"] == state["research_id"]
        assert retained["unresolved_questions"] == state["unresolved_questions"]
        assert retained["search_count"] == 7
        assert retained["failed_attempt_count"] == 2
        assert retained["remaining_budget"] == 5
        assert retained["telemetry"]["pending_recommendation"] == {
            "id": 4,
            "action": "continue_search",
            "searches_before": 7,
        }
        assert retained["telemetry"]["followed_recommendations"] == 2
        assert retained["telemetry"]["recommendation_count"] == 4
        nested, _ = _truncate_total(
            {"results": [{"research_state": state}], "message": "current"}, 1000
        )
        assert len(json.dumps(nested, ensure_ascii=False)) <= 1000
        assert nested["results"][0]["research_state"]["research_id"] == "workstream-1"

        latest = await plugin.after_tool_callback(
            tool=_tool("get_next_action"),
            tool_args={},
            tool_context=_tool_context(),
            result=oversized,
        )
        assert latest is not None
        assert len(json.dumps(latest, ensure_ascii=False)) <= 1000
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_function_response(
                    name="search",
                    response={"url": f"https://source.example/{index}", "content": "q" * 900},
                )],
            )
            for index in range(4)
        ]
        contents.append(types.Content(
            role="user",
            parts=[types.Part.from_function_response(name="get_next_action", response=latest)],
        ))
        request = LlmRequest(contents=contents)
        await plugin.before_model_callback(
            callback_context=SimpleNamespace(), llm_request=request
        )
        active = [
            part.function_response.response
            for content in request.contents
            for part in content.parts or []
            if part.function_response is not None
        ]
        assert sum(len(json.dumps(item, ensure_ascii=False)) for item in active) <= 1200
        active_state = next(item["research_state"] for item in active if "research_state" in item)
        assert active_state["telemetry"]["pending_recommendation"]["action"] == "continue_search"

        root = Path(__file__).resolve().parents[4]
        controller_path = (
            root / "mcp-servers/research-controller/src/mcp_research_controller/controller.py"
        )
        spec = importlib.util.spec_from_file_location("research_controller_for_test", controller_path)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(module)
        controller = module.ResearchController()
        updated = controller.update_research_state(
            goal=active_state["goal"],
            research_id=active_state["research_id"],
            research_state=active_state,
            unresolved_questions=active_state["unresolved_questions"],
            last_recommendation_followed=True,
        )["research_state"]
        assert controller.get_next_action(updated)["research_state"]["goal"] == state["goal"]

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
