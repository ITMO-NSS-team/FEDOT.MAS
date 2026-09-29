"""Execution budgets belong to one request, including all its loop iterations."""

from unittest.mock import patch

import pytest
from fedotmas.maw import builder
from fedotmas.maw.models import MAWConfig
from google.adk import Runner
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.sessions import DatabaseSessionService, InMemorySessionService
from google.adk.tools import FunctionTool
from google.genai import types


class ScriptedLlm(BaseLlm):
    responses: list[types.Content]
    calls: int = 0

    async def generate_content_async(self, llm_request, stream=False):
        self.calls += 1
        yield LlmResponse(content=self.responses.pop(0))


@pytest.mark.asyncio
@pytest.mark.parametrize("persistent", [False, True])
@pytest.mark.parametrize("loop", [False, True])
async def test_new_request_gets_new_budget_but_loop_iterations_share_it(
    tmp_path, persistent, loop
):
    node = {"type": "agent", "agent_name": "answerer"}
    config = MAWConfig(
        agents=[
            {
                "name": "answerer",
                "instruction": "Answer.",
                "output_key": "answer",
                "max_llm_turns": 1,
            }
        ],
        pipeline={"type": "loop", "max_iterations": 2, "children": [node]}
        if loop
        else node,
    )
    llm = ScriptedLlm(
        model="openai/test",
        responses=[
            types.Content(
                role="model",
                parts=[types.Part.from_function_call(name="ping", args={})],
            ),
            types.Content(role="model", parts=[types.Part.from_text(text="42")]),
        ],
    )
    with patch.object(builder, "_resolve_llm", return_value=llm):
        root = builder.build(config, autonomous=False)
    worker = root.sub_agents[0] if loop else root

    def ping() -> str:
        return "ok"

    worker.tools.append(FunctionTool(ping))
    service = (
        DatabaseSessionService(db_url=f"sqlite+aiosqlite:///{tmp_path / 'sessions.db'}")
        if persistent
        else InMemorySessionService()
    )
    session = await service.create_session(
        app_name="test", user_id="user", state={"kept": "artifact"}
    )
    invocation_ids = []
    async with Runner(agent=root, app_name="test", session_service=service) as runner:
        for index in range(2):
            async for _ in runner.run_async(
                user_id="user",
                session_id=session.id,
                new_message=types.Content(
                    role="user", parts=[types.Part.from_text(text=f"Question {index}")]
                ),
            ):
                pass
            current = await service.get_session(
                app_name="test", user_id="user", session_id=session.id
            )
            metadata = current.state["_fedotmas_execution"]
            invocation_ids.append(metadata["invocation_id"])
            assert metadata["agent_llm_turns"] == {"answerer": 1}
            assert current.state["kept"] == "artifact"
            assert llm.calls == index + 1
            if index == 0 or loop:
                assert "answerer" in metadata["limited_agents"]
            else:
                assert not metadata.get("limited_agents")
                assert current.state["answer"] == "42"
    assert invocation_ids[0] != invocation_ids[1]
