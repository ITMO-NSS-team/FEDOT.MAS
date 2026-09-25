from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest
from fedotmas.core.runner import run_pipeline
from fedotmas.maw import builder
from fedotmas.maw.models import MAWAgentConfig, MAWConfig, MAWStepConfig
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.sessions import InMemorySessionService
from google.genai import types
from pydantic import Field

CONTRACT = "Return the final answer inside <solution>...</solution>."


def _config(*, finalizer: bool, final_answer_agent: str | None = None) -> MAWConfig:
    agents = [
        MAWAgentConfig(
            name="writer",
            instruction="Write a draft for {user_query}. Revise using {feedback}.",
            output_key="draft",
        ),
        MAWAgentConfig(
            name="critic",
            instruction="Review {draft}; return feedback or call exit_loop.",
            output_key="feedback",
        ),
    ]
    loop = MAWStepConfig(
        type="loop",
        max_iterations=1,
        children=[
            MAWStepConfig(type="agent", agent_name="writer"),
            MAWStepConfig(type="agent", agent_name="critic"),
        ],
    )
    if finalizer:
        agents.append(
            MAWAgentConfig(
                name="answerer",
                instruction="Answer from final draft {draft} and critique {feedback}.",
                output_key="answer",
            )
        )
        pipeline = MAWStepConfig(
            type="sequential",
            children=[loop, MAWStepConfig(type="agent", agent_name="answerer")],
        )
    else:
        pipeline = loop
    return MAWConfig(
        agents=agents,
        pipeline=pipeline,
        final_answer_agent=final_answer_agent,
    )


def test_loop_participant_cannot_be_configured_as_final_answer_agent() -> None:
    config = _config(finalizer=False, final_answer_agent="critic")
    with pytest.raises(ValueError, match="add a post-loop finalizer"):
        builder.build(config, final_answer_contract=CONTRACT)


def test_bare_loop_cannot_infer_final_answer_agent() -> None:
    config = _config(finalizer=False)
    with pytest.raises(ValueError, match="add a post-loop finalizer"):
        builder.build(config, final_answer_contract=CONTRACT)
    assert config.final_answer_agent is None


def test_loop_without_external_contract_remains_buildable() -> None:
    config = _config(finalizer=False, final_answer_agent="critic")
    assert builder.build(config).sub_agents[-1].name == "critic"


@pytest.mark.parametrize("sequential", [False, True])
def test_non_loop_terminal_inference_is_unchanged(sequential: bool) -> None:
    answerer = MAWStepConfig(type="agent", agent_name="answerer")
    pipeline = (
        MAWStepConfig(
            type="sequential",
            children=[MAWStepConfig(type="agent", agent_name="writer"), answerer],
        )
        if sequential
        else answerer
    )
    agents = [
        MAWAgentConfig(name="answerer", instruction="Answer.", output_key="answer")
    ]
    if sequential:
        agents.insert(
            0, MAWAgentConfig(name="writer", instruction="Draft.", output_key="draft")
        )
    config = MAWConfig(agents=agents, pipeline=pipeline)

    builder.build(config, final_answer_contract=CONTRACT)

    assert config.final_answer_agent == "answerer"


class _ScriptedLlm(BaseLlm):
    responses: list[str]
    requests: list[LlmRequest] = Field(default_factory=list)

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        del stream
        self.requests.append(llm_request)
        yield LlmResponse(
            content=types.Content(
                role="model", parts=[types.Part.from_text(text=self.responses.pop(0))]
            )
        )


@pytest.mark.parametrize("explicit", [True, False])
@pytest.mark.asyncio
async def test_post_loop_finalizer_alone_receives_contract_and_final_state(
    monkeypatch: pytest.MonkeyPatch, explicit: bool
) -> None:
    config = _config(finalizer=True, final_answer_agent="answerer" if explicit else None)
    llm = _ScriptedLlm(
        model="openai/test",
        responses=["revised draft", "validated", "<solution>answer</solution>"],
    )
    monkeypatch.setattr(builder, "_resolve_llm", lambda *_args: llm)

    agent = builder.build(config, autonomous=False, final_answer_contract=CONTRACT)
    result = await run_pipeline(
        agent, "question", session_service=InMemorySessionService()
    )

    assert config.final_answer_agent == "answerer"
    assert result.status == "completed"
    assert result.state["draft"] == "revised draft"
    assert result.state["feedback"] == "validated"
    assert result.state["answer"] == "<solution>answer</solution>"
    assert len(llm.requests) == 3
    instructions = [request.config.system_instruction for request in llm.requests]
    assert all(CONTRACT not in instruction for instruction in instructions[:2])
    assert CONTRACT in instructions[2]
    assert "revised draft" in instructions[1]
    assert "revised draft" in instructions[2]
    assert "validated" in instructions[2]
