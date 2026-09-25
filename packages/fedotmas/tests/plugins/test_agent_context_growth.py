from __future__ import annotations

import json
from collections.abc import AsyncGenerator

import pytest
from fedotmas.core.runner import run_pipeline
from fedotmas.plugins import ToolResultTruncationPlugin
from google.adk.agents import LlmAgent
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


def _call(number: int) -> types.Content:
    return types.Content(
        role="model",
        parts=[
            types.Part(
                function_call=types.FunctionCall(
                    id=f"read-{number}", name="read_page", args={"page": number}
                )
            )
        ],
    )


def _answer() -> types.Content:
    return types.Content(role="model", parts=[types.Part.from_text(text="Done.")])


def _request_chars(request: LlmRequest) -> int:
    return len(
        json.dumps(
            request.model_dump(mode="json", exclude={"tools_dict"}),
            ensure_ascii=False,
            default=str,
        )
    )


async def _run_repeated_pages(*, compact: bool) -> list[int]:
    def read_page(page: int) -> dict[str, str | int]:
        """Read one synthetic source page."""
        return {
            "url": f"https://example.org/page/{page}",
            "title": f"Source page {page}",
            "content": (f"Page {page} evidence. " + "x" * 30000),
        }

    responses = [_call(number) for number in range(6)] + [_answer()]
    llm = _ScriptedLlm(model="test", responses=responses)
    agent = LlmAgent(
        name="researcher",
        model=llm,
        instruction="Inspect the pages for {user_query}.",
        tools=[FunctionTool(read_page)],
        output_key="result",
        include_contents="none",
    )
    plugins = (
        [
            ToolResultTruncationPlugin(
                max_string_chars=5000,
                max_total_chars=3000,
                aggregate_tool_names={"*"},
                max_agent_total_chars=9000,
            )
        ]
        if compact
        else []
    )
    await run_pipeline(
        agent,
        "Keep each evidence excerpt and source URL.",
        session_service=InMemorySessionService(),
        plugins=plugins,
    )
    return [_request_chars(request) for request in llm.requests]


@pytest.mark.asyncio
async def test_large_tool_outputs_have_bounded_active_prompt_growth():
    baseline_chars = await _run_repeated_pages(compact=False)
    bounded_chars = await _run_repeated_pages(compact=True)
    baseline_tokens = [chars // 4 for chars in baseline_chars]
    bounded_tokens = [chars // 4 for chars in bounded_chars]

    print(
        "synthetic prompt estimate (chars/4): "
        f"before={baseline_tokens}; after={bounded_tokens}"
    )
    assert max(bounded_chars) < max(baseline_chars) // 5
    assert max(bounded_chars) < 25000
    assert max(bounded_tokens) < 6250
