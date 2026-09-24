from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fedotmas.maw.maw import MAW
from fedotmas.plugins import ToolErrorCircuitBreakerPlugin, WebSearchLimitPlugin
from google.adk.agents import LlmAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.adk.flows.llm_flows.functions import handle_function_calls_async
from google.adk.plugins.plugin_manager import PluginManager
from google.adk.sessions import InMemorySessionService
from google.adk.sessions.session import Session
from google.adk.tools import FunctionTool
from google.genai import types


def _tool(name: str, description: str = "") -> MagicMock:
    tool = MagicMock()
    tool.name = name
    tool.description = description
    return tool


def _tool_context(*, session_id: str = "s1", agent_name: str = "researcher"):
    ctx = MagicMock()
    ctx._invocation_context.session.id = session_id
    ctx._invocation_context.agent.name = agent_name
    return ctx


def _invocation_context(session_id: str = "s1"):
    ctx = MagicMock()
    ctx.session.id = session_id
    return ctx


class TestWebSearchLimitPlugin:
    @pytest.mark.asyncio
    async def test_scraping_has_its_own_limit(self):
        search_limit = WebSearchLimitPlugin(max_calls_per_agent=1)
        scrape_limit = WebSearchLimitPlugin(
            max_calls_per_agent=1, tool_names={"goto"}, budget_kind="scraping"
        )
        ctx = _tool_context()
        search = _tool("search", "Search the web")
        goto = _tool("goto")
        await search_limit.before_tool_callback(
            tool=search, tool_args={"query": "first"}, tool_context=ctx
        )
        assert (
            await search_limit.before_tool_callback(
                tool=search, tool_args={"query": "second"}, tool_context=ctx
            )
        )["error_code"] == "WEB_BUDGET_EXHAUSTED"
        assert (
            await scrape_limit.before_tool_callback(
                tool=goto, tool_args={"url": "https://example.com/a"}, tool_context=ctx
            )
            is None
        )
        assert (
            await scrape_limit.before_tool_callback(
                tool=goto, tool_args={"url": "https://example.com/b"}, tool_context=ctx
            )
        )["error_code"] == "WEB_BUDGET_EXHAUSTED"

    @pytest.mark.asyncio
    async def test_duplicate_call_does_not_execute_backend_or_trip_breaker(self):
        calls: list[str] = []

        def search(query: str) -> dict[str, str]:
            """Search the web."""
            calls.append(query)
            return {"query": query}

        agent = LlmAgent(name="researcher", model="gemini-2.0-flash")
        context = InvocationContext(
            invocation_id="invocation",
            session_service=InMemorySessionService(),
            session=Session(id="session", app_name="test", user_id="user"),
            agent=agent,
            plugin_manager=PluginManager(
                [
                    WebSearchLimitPlugin(max_calls_per_agent=2),
                    ToolErrorCircuitBreakerPlugin(
                        max_errors_per_agent=1, max_same_tool_error_type=1
                    ),
                ]
            ),
        )
        calls_requested = [
            types.FunctionCall(
                id=f"call-{index}", name="search", args={"query": "same"}
            )
            for index in range(2)
        ]
        event = await handle_function_calls_async(
            context,
            Event(
                invocation_id="invocation",
                author="researcher",
                content=types.Content(
                    role="model",
                    parts=[types.Part(function_call=call) for call in calls_requested],
                ),
            ),
            {"search": FunctionTool(search)},
        )
        assert calls == ["same"]
        responses = [part.function_response.response for part in event.content.parts]
        assert len(responses) == 2
        assert sum(r.get("error_code") == "DUPLICATE_TOOL_CALL" for r in responses) == 1

    @pytest.mark.asyncio
    async def test_allows_until_limit_then_blocks(self):
        plugin = WebSearchLimitPlugin(max_calls_per_agent=2)
        tool = _tool("search", "Search the web")
        ctx = _tool_context()

        first = await plugin.before_tool_callback(
            tool=tool, tool_args={"query": "a"}, tool_context=ctx
        )
        second = await plugin.before_tool_callback(
            tool=tool, tool_args={"query": "b"}, tool_context=ctx
        )
        third = await plugin.before_tool_callback(
            tool=tool, tool_args={"query": "c"}, tool_context=ctx
        )

        assert first is None
        assert second is None
        assert third is not None
        assert third["isError"] is True
        assert "max 2 calls" in third["error"]

    @pytest.mark.asyncio
    async def test_search_exhaustion_allows_scraping_and_sibling_work(self):
        search_limit = WebSearchLimitPlugin(
            max_calls_per_agent=1,
        )
        scrape_limit = WebSearchLimitPlugin(
            max_calls_per_agent=2,
            tool_names={"goto"},
        )
        search = _tool("search", "Search the web")
        goto = _tool("goto")

        assert (
            await search_limit.before_tool_callback(
                tool=search, tool_args={"query": "first"}, tool_context=_tool_context()
            )
            is None
        )
        exhausted = await search_limit.before_tool_callback(
            tool=search,
            tool_args={"query": "over budget"},
            tool_context=_tool_context(),
        )
        blocked_scrape = await scrape_limit.before_tool_callback(
            tool=goto,
            tool_args={"url": "https://example.com"},
            tool_context=_tool_context(),
        )
        sibling_scrape = await scrape_limit.before_tool_callback(
            tool=goto,
            tool_args={"url": "https://example.com"},
            tool_context=_tool_context(agent_name="sibling"),
        )

        assert exhausted is not None and exhausted["isError"] is True
        assert "Inspect already-found URLs" in exhausted["error"]
        assert exhausted["error_code"] == "WEB_BUDGET_EXHAUSTED"
        assert blocked_scrape is None
        assert sibling_scrape is None

    @pytest.mark.asyncio
    async def test_counts_are_per_agent(self):
        plugin = WebSearchLimitPlugin(max_calls_per_agent=1)
        tool = _tool("web_search")

        await plugin.before_tool_callback(
            tool=tool, tool_args={}, tool_context=_tool_context(agent_name="a")
        )
        result = await plugin.before_tool_callback(
            tool=tool, tool_args={}, tool_context=_tool_context(agent_name="b")
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_parallel_calls_all_receive_correlated_results_when_budget_expires(
        self,
    ):
        def search(query: str) -> dict[str, str]:
            """Search the web."""
            return {"query": query}

        plugin = WebSearchLimitPlugin(max_calls_per_agent=1)
        agent = LlmAgent(name="researcher", model="gemini-2.0-flash")
        invocation_context = InvocationContext(
            invocation_id="invocation",
            session_service=InMemorySessionService(),
            session=Session(id="session", app_name="test", user_id="user"),
            agent=agent,
            plugin_manager=PluginManager(
                [
                    plugin,
                    ToolErrorCircuitBreakerPlugin(
                        max_errors_per_agent=1,
                        max_same_tool_error_type=1,
                    ),
                ]
            ),
        )
        calls = [
            types.FunctionCall(
                id=f"call-{index}", name="search", args={"query": f"q{index}"}
            )
            for index in range(3)
        ]
        response_event = await handle_function_calls_async(
            invocation_context,
            Event(
                invocation_id="invocation",
                author="researcher",
                content=types.Content(
                    role="model",
                    parts=[types.Part(function_call=call) for call in calls],
                ),
            ),
            {"search": FunctionTool(search)},
        )

        assert response_event is not None
        responses = {
            part.function_response.id: part.function_response.response
            for part in response_event.content.parts
            if part.function_response is not None
        }
        assert set(responses) == {call.id for call in calls}
        assert sum("isError" in response for response in responses.values()) == 2
        assert (
            sum(
                response.get("error_code") == "WEB_BUDGET_EXHAUSTED"
                for response in responses.values()
            )
            == 2
        )
        assert sum("error_code" not in response for response in responses.values()) == 1

    @pytest.mark.asyncio
    async def test_hard_fail_true_is_deprecated_and_still_returns_a_result(self):
        with pytest.warns(DeprecationWarning, match="hard_fail=True is deprecated"):
            plugin = WebSearchLimitPlugin(max_calls_per_agent=1, hard_fail=True)
        tool = _tool("search", "Search the web")
        ctx = _tool_context()

        assert (
            await plugin.before_tool_callback(
                tool=tool, tool_args={"query": "first"}, tool_context=ctx
            )
            is None
        )
        blocked = await plugin.before_tool_callback(
            tool=tool, tool_args={"query": "second"}, tool_context=ctx
        )

        assert blocked is not None
        assert blocked["error_code"] == "WEB_BUDGET_EXHAUSTED"

    @pytest.mark.asyncio
    async def test_ignores_plain_non_web_search_tool(self):
        plugin = WebSearchLimitPlugin(max_calls_per_agent=1)
        tool = _tool("search", "Search local memory")
        ctx = _tool_context()

        await plugin.before_tool_callback(tool=tool, tool_args={}, tool_context=ctx)
        result = await plugin.before_tool_callback(
            tool=tool, tool_args={}, tool_context=ctx
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_before_run_resets_current_session_counts(self):
        plugin = WebSearchLimitPlugin(max_calls_per_agent=1)
        tool = _tool("search", "Search the internet")
        ctx = _tool_context(session_id="s1")

        await plugin.before_tool_callback(tool=tool, tool_args={}, tool_context=ctx)
        exhausted = await plugin.before_tool_callback(
            tool=tool, tool_args={"query": "over budget"}, tool_context=ctx
        )
        assert exhausted is not None
        await plugin.before_run_callback(invocation_context=_invocation_context("s1"))
        result = await plugin.before_tool_callback(
            tool=tool, tool_args={}, tool_context=ctx
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_dedupes_identical_calls(self):
        plugin = WebSearchLimitPlugin(max_calls_per_agent=1)
        tool = _tool("search", "Search the internet")
        ctx = _tool_context()

        first = await plugin.before_tool_callback(
            tool=tool, tool_args={"query": "same"}, tool_context=ctx
        )
        duplicate = await plugin.before_tool_callback(
            tool=tool, tool_args={"query": "same"}, tool_context=ctx
        )
        over_limit = await plugin.before_tool_callback(
            tool=tool, tool_args={"query": "new"}, tool_context=ctx
        )

        assert first is None
        assert duplicate is not None
        assert duplicate["error_code"] == "DUPLICATE_TOOL_CALL"
        assert over_limit is not None
        assert over_limit["isError"] is True

    @pytest.mark.asyncio
    async def test_can_count_unique_remote_urls(self):
        plugin = WebSearchLimitPlugin(
            max_calls_per_agent=1,
            tool_names={"goto", "markdown"},
            count_unique_urls=True,
        )
        ctx = _tool_context()

        first = await plugin.before_tool_callback(
            tool=_tool("goto"),
            tool_args={"url": "https://example.com/path?b=2&a=1#frag"},
            tool_context=ctx,
        )
        same_url = await plugin.before_tool_callback(
            tool=_tool("markdown"),
            tool_args={"url": "https://example.com/path?a=1&b=2"},
            tool_context=ctx,
        )
        over_limit = await plugin.before_tool_callback(
            tool=_tool("goto"),
            tool_args={"url": "https://example.org/"},
            tool_context=ctx,
        )

        assert first is None
        assert same_url is None
        assert over_limit is not None
        assert over_limit["isError"] is True

    @pytest.mark.asyncio
    async def test_ignores_local_urls_and_rejects_empty_urls(self):
        plugin = WebSearchLimitPlugin(
            max_calls_per_agent=1,
            tool_names={"goto"},
            reject_empty_urls=True,
        )
        ctx = _tool_context()

        local = await plugin.before_tool_callback(
            tool=_tool("goto"),
            tool_args={"url": "file:///tmp/input.mp3"},
            tool_context=ctx,
        )
        empty = await plugin.before_tool_callback(
            tool=_tool("goto"), tool_args={"url": ""}, tool_context=ctx
        )
        remote = await plugin.before_tool_callback(
            tool=_tool("goto"),
            tool_args={"url": "https://example.com/"},
            tool_context=ctx,
        )

        assert local is None
        assert empty is not None
        assert "Empty URL" in empty["error"]
        assert remote is None


class TestAutoAddWebSearchLimitPlugin:
    def test_default_has_web_search_limit_plugin(self):
        maw = MAW()
        plugin = next(p for p in maw._plugins if isinstance(p, WebSearchLimitPlugin))
        assert plugin.max_calls_per_agent == 20

    def test_can_disable_default_web_search_limit_plugin(self):
        maw = MAW(web_search_limit=None)
        assert not any(isinstance(p, WebSearchLimitPlugin) for p in maw._plugins)
