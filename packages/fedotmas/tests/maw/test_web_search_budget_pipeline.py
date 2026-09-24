"""End-to-end regressions for per-agent web-search budgets in ADK pipelines."""

from __future__ import annotations

from collections.abc import AsyncGenerator

from fedotmas.core.runner import run_pipeline
from fedotmas.plugins import (
    ToolErrorCircuitBreakerPlugin,
    WebSearchLimitPlugin,
)
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool
from google.genai import types
from pydantic import Field


class _ScriptedLlm(BaseLlm):
    responses: list[types.Content]
    requests: list[LlmRequest] = Field(default_factory=list)

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        del stream
        self.requests.append(llm_request)
        yield LlmResponse(content=self.responses.pop(0))


def _call(query: str, call_id: str) -> types.Content:
    return types.Content(
        role="model",
        parts=[
            types.Part(
                function_call=types.FunctionCall(
                    id=call_id,
                    name="search",
                    args={"query": query},
                )
            )
        ],
    )


def _answer(text: str) -> types.Content:
    return types.Content(role="model", parts=[types.Part.from_text(text=text)])


def _function_parts(requests: list[LlmRequest]):
    for request in requests:
        for content in request.contents:
            for part in content.parts or []:
                if part.function_call is not None:
                    yield "call", part.function_call
                elif part.function_response is not None:
                    yield "response", part.function_response


async def test_sequential_agents_keep_budget_exhaustion_local():
    """Agent A finishes after exhaustion; B gets a fresh budget and runs."""
    actual_searches: list[str] = []

    def search(query: str) -> dict[str, str]:
        """Search the web for evidence matching the query."""
        actual_searches.append(query)
        return {"query": query, "evidence": f"evidence for {query}"}

    agent_a_llm = _ScriptedLlm(
        model="test",
        responses=[
            _call("initial evidence", "a-call-1"),
            _call("blocked after exhaustion", "a-call-2"),
            _call("blocked again after exhaustion", "a-call-3"),
            _answer("Agent A finished from gathered evidence."),
        ],
    )
    agent_b_llm = _ScriptedLlm(
        model="test",
        responses=[
            _call("agent B independent search", "b-call-1"),
            _answer("Agent B finished."),
        ],
    )
    agent_a = LlmAgent(
        name="agent_a",
        instruction="Research {user_query} and return the supported result.",
        model=agent_a_llm,
        tools=[FunctionTool(search)],
        output_key="agent_a_output",
        include_contents="none",
    )
    agent_b = LlmAgent(
        name="agent_b",
        instruction="Use {agent_a_output} to continue the task {user_query}.",
        model=agent_b_llm,
        tools=[FunctionTool(search)],
        output_key="agent_b_output",
        include_contents="none",
    )
    pipeline = SequentialAgent(name="pipeline", sub_agents=[agent_a, agent_b])
    web_limit = WebSearchLimitPlugin(max_calls_per_agent=1)
    circuit_breaker = ToolErrorCircuitBreakerPlugin(
        max_errors_per_agent=1,
        max_same_tool_error_type=1,
    )

    result = await run_pipeline(
        pipeline,
        "Find and verify a general public fact.",
        session_service=InMemorySessionService(),
        plugins=[web_limit, circuit_breaker],
    )

    assert actual_searches == ["initial evidence", "agent B independent search"]
    assert result.state["agent_a_output"] == "Agent A finished from gathered evidence."
    assert result.state["agent_b_output"] == "Agent B finished."

    calls: dict[str, str] = {}
    responses: dict[str, dict] = {}
    final_request = agent_a_llm.requests[-1]
    for kind, part in _function_parts([final_request]):
        if kind == "call" and part.id is not None:
            calls[part.id] = str(part.args["query"])
        elif kind == "response" and part.id is not None:
            responses[part.id] = part.response

    assert calls == {
        "a-call-1": "initial evidence",
        "a-call-2": "blocked after exhaustion",
        "a-call-3": "blocked again after exhaustion",
    }
    assert set(responses) == set(calls)
    assert [responses[call_id].get("error_code") for call_id in calls] == [
        None,
        "WEB_BUDGET_EXHAUSTED",
        "WEB_BUDGET_EXHAUSTED",
    ]


async def test_search_exhaustion_still_allows_evidence_extraction():
    searched: list[str] = []
    inspected: list[str] = []

    def search(query: str) -> dict[str, str]:
        """Search the web for sources."""
        searched.append(query)
        return {"url": "https://example.com/source"}

    def goto(url: str) -> dict[str, str]:
        """Inspect a discovered web page."""
        inspected.append(url)
        return {"evidence": "source text"}

    llm = _ScriptedLlm(
        model="test",
        responses=[
            _call("find source", "search-1"),
            _call("new discovery", "search-2"),
            types.Content(
                role="model",
                parts=[
                    types.Part(
                        function_call=types.FunctionCall(
                            id="inspect-1",
                            name="goto",
                            args={"url": "https://example.com/source"},
                        )
                    )
                ],
            ),
            _answer("Answer from inspected source."),
        ],
    )
    agent = LlmAgent(
        name="researcher",
        model=llm,
        instruction="Research {user_query}.",
        tools=[FunctionTool(search), FunctionTool(goto)],
        output_key="answer",
        include_contents="none",
    )
    result = await run_pipeline(
        agent,
        "Question",
        session_service=InMemorySessionService(),
        plugins=[
            WebSearchLimitPlugin(max_calls_per_agent=1),
            WebSearchLimitPlugin(
                max_calls_per_agent=1,
                tool_names={"goto"},
                budget_kind="scraping",
                name="scraping_limit",
            ),
            ToolErrorCircuitBreakerPlugin(
                max_errors_per_agent=1, max_same_tool_error_type=1
            ),
        ],
    )
    assert searched == ["find source"]
    assert inspected == ["https://example.com/source"]
    assert result.state["answer"] == "Answer from inspected source."
