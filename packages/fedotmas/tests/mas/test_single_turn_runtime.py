"""Deterministic ADK runtime coverage for MAS single-turn delegation."""

from __future__ import annotations

from collections.abc import AsyncGenerator

from google.adk import Runner
from google.adk.agents import LlmAgent
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.sessions import InMemorySessionService
from google.genai import types
from pydantic import Field


class _ScriptedLlm(BaseLlm):
    """Returns one prepared ADK response per model turn and records requests."""

    responses: list[types.Content]
    requests: list[LlmRequest] = Field(default_factory=list)

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        del stream
        self.requests.append(llm_request)
        yield LlmResponse(content=self.responses.pop(0))


def _function_call(name: str, request: str) -> types.Content:
    return types.Content(
        role="model",
        parts=[types.Part.from_function_call(name=name, args={"request": request})],
    )


def _text(text: str) -> types.Content:
    return types.Content(role="model", parts=[types.Part.from_text(text=text)])


async def test_single_turn_workers_return_control_and_receive_session_state():
    coordinator_llm = _ScriptedLlm(
        model="test",
        responses=[
            _function_call("worker1", "Calculate using N=20."),
            _function_call("worker2", "Verify worker1's result."),
            _text("Final verified answer."),
        ],
    )
    worker1_llm = _ScriptedLlm(model="test", responses=[_text("calculation")])
    worker2_llm = _ScriptedLlm(model="test", responses=[_text("verified")])
    worker1 = LlmAgent(
        name="worker1",
        description="Calculates",
        instruction="Calculate N={n}.",
        model=worker1_llm,
        mode="single_turn",
        output_key="worker1_output",
    )
    worker2 = LlmAgent(
        name="worker2",
        description="Verifies",
        instruction="Verify N={n}.",
        model=worker2_llm,
        mode="single_turn",
        output_key="worker2_output",
    )
    coordinator = LlmAgent(
        name="coordinator",
        instruction="Call worker1, then worker2, then answer.",
        model=coordinator_llm,
        sub_agents=[worker1, worker2],
    )
    sessions = InMemorySessionService()
    session = await sessions.create_session(
        app_name="single-turn-test",
        user_id="user",
        session_id="session",
        state={"n": 20},
    )
    runner = Runner(
        app_name="single-turn-test", agent=coordinator, session_service=sessions
    )

    events = [
        event
        async for event in runner.run_async(
            user_id="user",
            session_id=session.id,
            new_message=types.Content(
                role="user", parts=[types.Part.from_text(text="Run the task.")]
            ),
        )
    ]

    assert [event.author for event in events] == [
        "coordinator",
        "worker1",
        "coordinator",
        "coordinator",
        "worker2",
        "coordinator",
        "coordinator",
    ]
    assert len(coordinator_llm.requests) == 3
    worker1_instruction = worker1_llm.requests[0].config.system_instruction
    worker2_instruction = worker2_llm.requests[0].config.system_instruction
    assert isinstance(worker1_instruction, str)
    assert isinstance(worker2_instruction, str)
    assert "N=20" in worker1_instruction
    assert "N=20" in worker2_instruction

    final_session = await sessions.get_session(
        app_name="single-turn-test", user_id="user", session_id=session.id
    )
    assert final_session is not None
    assert final_session.state["worker1_output"] == "calculation"
    assert final_session.state["worker2_output"] == "verified"
