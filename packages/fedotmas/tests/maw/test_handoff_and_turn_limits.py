from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from unittest.mock import MagicMock

import pytest
from fedotmas.core.runner import run_pipeline
from fedotmas.maw import builder
from fedotmas.maw.models import (
    ArtifactContract,
    ArtifactRequirement,
    MAWAgentConfig,
    MAWConfig,
    MAWStepConfig,
)
from fedotmas.plugins import WebSearchLimitPlugin
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool
from google.genai import types
from pydantic import Field


def _context(state: dict) -> MagicMock:
    context = MagicMock()
    context.state = state
    context._invocation_context.session.state = state
    context._invocation_context.artifact_service = None
    return context


def _contract_config() -> MAWConfig:
    return MAWConfig(
        agents=[
            MAWAgentConfig(
                name="paper_researcher",
                instruction="Find the requested paper and evidence.",
                output_key="paper_evidence",
                output_contract=ArtifactContract(
                    description="Identified paper and evidence",
                    required_fields=["paper_identity", "equation", "evidence"],
                    identity_fields=["paper_identity"],
                ),
            ),
            MAWAgentConfig(
                name="verifier",
                instruction="Verify the requested claim using {paper_evidence}.",
                output_key="verification",
                input_requirements=[
                    ArtifactRequirement(
                        source_key="paper_evidence",
                        required_fields=["paper_identity", "equation", "evidence"],
                        identity_fields=["paper_identity"],
                        purpose="Verify the equation for the selected paper.",
                    )
                ],
                output_contract=ArtifactContract(
                    required_fields=["paper_identity", "decision"],
                    identity_fields=["paper_identity"],
                ),
                research_policy="evidence_first",
            ),
        ],
        pipeline=MAWStepConfig(
            type="sequential",
            children=[
                MAWStepConfig(type="agent", agent_name="paper_researcher"),
                MAWStepConfig(type="agent", agent_name="verifier"),
            ],
        ),
    )


def test_nonterminal_identity_consumer_must_propagate_identity():
    config = _contract_config().model_copy(
        deep=True,
        update={
            "pipeline": MAWStepConfig(
                type="sequential",
                children=[
                    MAWStepConfig(type="agent", agent_name="paper_researcher"),
                    MAWStepConfig(type="agent", agent_name="verifier"),
                    MAWStepConfig(type="agent", agent_name="answerer"),
                ],
            ),
            "agents": [
                *[a.model_copy(deep=True) for a in _contract_config().agents],
                MAWAgentConfig(
                    name="answerer", instruction="Answer.", output_key="answer"
                ),
            ],
            "final_answer_agent": "answerer",
        },
    )
    config.agents[1].output_contract = None

    with pytest.raises(ValueError, match="must include upstream identity fields"):
        MAWConfig.model_validate(config.model_dump())


@pytest.mark.parametrize("explicit", [True, False])
def test_terminal_identity_consumer_may_use_external_answer_boundary(explicit):
    config = _contract_config().model_copy(
        deep=True,
        update={
            "final_answer_agent": "verifier" if explicit else None,
        },
    )
    config.agents[1].output_contract = None

    validated = MAWConfig.model_validate(config.model_dump())
    assert validated.agents[1].input_requirements[0].identity_fields == [
        "paper_identity"
    ]


def test_terminal_structured_identity_output_remains_valid():
    config = _contract_config().model_copy(
        update={"final_answer_agent": "verifier"}
    )

    validated = MAWConfig.model_validate(config.model_dump())
    assert validated.agents[1].output_contract.identity_fields == ["paper_identity"]


@pytest.mark.asyncio
async def test_structured_handoff_is_preserved_with_entity_identity():
    config = _contract_config()
    root = builder.build(config, autonomous=False)
    _producer, consumer = root.sub_agents
    artifact = json.dumps(
        {
            "paper_identity": "doi:10.5555/example",
            "equation": "E = mc^2",
            "evidence": "Equation 4, page 8",
            "sources": ["https://example.org/paper"],
        }
    )
    state = {"paper_evidence": artifact}

    text = await consumer.instruction(_context(state))
    assert "Equation 4, page 8" in text
    assert "https://example.org/paper" in text
    assert "ENTITY CONTINUITY" in text
    assert "doi:10.5555/example" in text
    assert "MISSING INPUT" not in text


@pytest.mark.asyncio
async def test_incomplete_and_scalar_handoffs_are_marked_and_retained():
    consumer = builder.build(_contract_config(), autonomous=False).sub_agents[1]

    missing_state = {
        "paper_evidence": json.dumps(
            {"paper_identity": "paper-A", "equation": "E=mc^2"}
        )
    }
    missing_text = await consumer.instruction(_context(missing_state))
    assert "INCOMPLETE HANDOFF" in missing_text
    assert "evidence" in missing_text
    assert "paper-A" in missing_text
    await consumer.before_agent_callback(_context(missing_state))
    issues = missing_state["_fedotmas_execution"]["handoff_issues"]
    assert issues[0]["kind"] == "incomplete_handoff"
    assert issues[0]["missing_fields"] == ["evidence"]

    identity_state = {
        "paper_evidence": json.dumps(
            {"paper_identity": " ", "equation": "E=mc^2", "evidence": "page 8"}
        )
    }
    identity_text = await consumer.instruction(_context(identity_state))
    assert "INCOMPLETE HANDOFF" in identity_text
    assert "paper_identity" in identity_text

    scalar = "<solution>50</solution>"
    scalar_state = {"paper_evidence": scalar}
    scalar_text = await consumer.instruction(_context(scalar_state))
    assert "INCOMPLETE HANDOFF" in scalar_text
    assert scalar in scalar_text
    await consumer.before_agent_callback(_context(scalar_state))
    assert scalar_state["_fedotmas_execution"]["handoff_issues"][0][
        "missing_fields"
    ] == ["paper_identity", "equation", "evidence"]


@pytest.mark.asyncio
async def test_downstream_identity_mismatch_is_recorded_as_incomplete():
    consumer = builder.build(_contract_config(), autonomous=False).sub_agents[1]
    state = {
        "paper_evidence": json.dumps(
            {
                "paper_identity": "paper-A",
                "equation": "E=mc^2",
                "evidence": "page 8",
            }
        ),
        "verification": json.dumps(
            {"paper_identity": "paper-B", "decision": "supported"}
        ),
    }

    await consumer.after_agent_callback(_context(state))

    issue = state["_fedotmas_execution"]["handoff_issues"][0]
    assert issue["kind"] == "entity_continuity_mismatch"
    assert issue["fields"] == ["paper_identity"]


@pytest.mark.asyncio
async def test_terminal_dependent_agent_also_preserves_required_identity():
    config = _contract_config().model_copy(update={"final_answer_agent": "verifier"})
    consumer = builder.build(config, autonomous=False).sub_agents[1]
    state = {
        "paper_evidence": json.dumps(
            {
                "paper_identity": "paper-A",
                "equation": "E=mc^2",
                "evidence": "page 8",
            }
        ),
        "verification": json.dumps(
            {"paper_identity": "paper-B", "decision": "supported"}
        ),
    }

    await consumer.after_agent_callback(_context(state))

    issue = state["_fedotmas_execution"]["handoff_issues"][0]
    assert issue["kind"] == "entity_continuity_mismatch"
    assert issue["fields"] == ["paper_identity"]


@pytest.mark.asyncio
async def test_complete_evidence_first_verifier_does_not_search(monkeypatch):
    backend_calls: list[str] = []

    def search(query: str) -> dict[str, str]:
        """Search for independent verification sources."""
        backend_calls.append(query)
        return {"query": query}

    def websearch_searxng_search(query: str) -> dict[str, str]:
        """Search the web with the SearXNG metasearch backend."""
        backend_calls.append(query)
        return {"query": query}

    config = _contract_config().model_copy(
        update={"pipeline": MAWStepConfig(type="agent", agent_name="verifier")}
    )
    llm = _ScriptedLlm(
        model="openai/test",
        responses=[
            types.Content(
                role="model",
                parts=[
                    types.Part.from_text(
                        text=json.dumps(
                            {"paper_identity": "paper-A", "decision": "supported"}
                        )
                    )
                ],
            )
        ],
    )
    monkeypatch.setattr(builder, "_resolve_llm", lambda *_args: llm)
    root = builder.build(config, autonomous=False)
    root.tools = [FunctionTool(search), FunctionTool(websearch_searxng_search)]
    evidence = json.dumps(
        {
            "paper_identity": "paper-A",
            "equation": "E=mc^2",
            "evidence": "Equation 4 on page 8",
        }
    )

    result = await run_pipeline(
        root,
        "Verify the equation.",
        session_service=InMemorySessionService(),
        initial_state={"paper_evidence": evidence},
    )

    declarations = [
        declaration
        for group in llm.requests[0].config.tools or []
        for declaration in group.function_declarations or []
    ]
    assert backend_calls == []
    assert not any(declaration.name == "search" for declaration in declarations)
    assert not any(
        declaration.name == "websearch_searxng_search" for declaration in declarations
    )
    assert "Equation 4 on page 8" in llm.requests[0].config.system_instruction
    assert result.status == "completed"


@pytest.mark.asyncio
async def test_final_contract_is_added_only_to_configured_terminal_agent():
    config = MAWConfig(
        agents=[
            MAWAgentConfig(
                name="researcher",
                instruction="Return a structured evidence packet.",
                output_key="findings",
            ),
            MAWAgentConfig(
                name="answerer",
                instruction="Answer from {findings}.",
                output_key="answer",
            ),
        ],
        pipeline=MAWStepConfig(
            type="sequential",
            children=[
                MAWStepConfig(type="agent", agent_name="researcher"),
                MAWStepConfig(type="agent", agent_name="answerer"),
            ],
        ),
        final_answer_agent="answerer",
    )
    root = builder.build(
        config,
        autonomous=False,
        final_answer_contract="Use <solution>...</solution> with only the answer.",
    )
    researcher, answerer = root.sub_agents

    assert "<solution>" not in researcher.instruction
    terminal_text = await answerer.instruction(
        _context({"findings": "structured evidence"})
    )
    assert "<solution>...</solution>" in terminal_text


@pytest.mark.asyncio
async def test_legacy_maw_config_without_contracts_remains_buildable():
    config = MAWConfig(
        agents=[
            MAWAgentConfig(
                name="solver",
                instruction="Answer the task in {user_query}.",
                output_key="answer",
            )
        ],
        pipeline=MAWStepConfig(agent_name="solver"),
    )

    agent = builder.build(config, autonomous=False)

    assert agent.name == "solver"
    assert agent.include_contents == "none"
    text = await agent.instruction(_context({"user_query": "legacy task"}))
    assert "FINAL ANSWER CONTRACT" not in text


@pytest.mark.asyncio
async def test_terminal_final_answer_skips_generated_output_contract(monkeypatch):
    config = MAWConfig(
        agents=[
            MAWAgentConfig(
                name="answerer",
                instruction="Answer the question.",
                output_key="answer",
                output_contract=ArtifactContract(required_fields=["answer"]),
            )
        ],
        pipeline=MAWStepConfig(type="agent", agent_name="answerer"),
        final_answer_agent="answerer",
    )
    llm = _ScriptedLlm(
        model="openai/test",
        responses=[types.Content(role="model", parts=[types.Part.from_text(text="<solution>42</solution>")])],
    )
    monkeypatch.setattr(builder, "_resolve_llm", lambda *_args: llm)
    agent = builder.build(
        config,
        autonomous=False,
        final_answer_contract="Use <solution>...</solution> with only the answer.",
    )

    result = await run_pipeline(
        agent, "Give the answer.", session_service=InMemorySessionService()
    )

    assert result.status == "completed"
    assert result.state["answer"] == "<solution>42</solution>"
    assert "handoff_issues" not in result.state.get("_fedotmas_execution", {})


@pytest.mark.parametrize(
    ("policy", "completed"),
    [("targeted_recovery", True), ("evidence_first", False), ("independent", False)],
)
@pytest.mark.asyncio
async def test_terminal_targeted_recovery_resolves_missing_handoff(monkeypatch, policy, completed):
    config = MAWConfig(
        agents=[
            MAWAgentConfig(name="producer", instruction="Return evidence.", output_key="evidence"),
            MAWAgentConfig(
                name="answerer",
                instruction="Recover the missing claim and answer.",
                output_key="answer",
                input_requirements=[ArtifactRequirement(source_key="evidence", required_fields=["claim"])],
                output_contract=ArtifactContract(required_fields=["claim"]),
                research_policy=policy,
            ),
        ],
        pipeline=MAWStepConfig(
            type="sequential",
            children=[MAWStepConfig(type="agent", agent_name="producer"), MAWStepConfig(type="agent", agent_name="answerer")],
        ),
        final_answer_agent="answerer",
    )
    llm = _ScriptedLlm(
        model="openai/test",
        responses=[
            types.Content(role="model", parts=[types.Part.from_text(text='{"other":"x"}')]),
            types.Content(role="model", parts=[types.Part.from_text(text="<solution>42</solution>")]),
        ],
    )
    monkeypatch.setattr(builder, "_resolve_llm", lambda *_args: llm)
    result = await run_pipeline(
        builder.build(config, autonomous=False, final_answer_contract="Use <solution>answer</solution>."),
        "Recover and answer.",
        session_service=InMemorySessionService(),
    )

    issue = result.state["_fedotmas_execution"]["handoff_issues"][0]
    assert issue["kind"] == "incomplete_handoff"
    assert issue["resolved"] is completed
    assert result.state["answer"] == "<solution>42</solution>"
    assert result.status == ("completed" if completed else "incomplete")


@pytest.mark.asyncio
async def test_inferred_final_answer_agent_is_persisted(monkeypatch):
    config = MAWConfig(
        agents=[MAWAgentConfig(name="answerer", instruction="Answer.", output_key="answer")],
        pipeline=MAWStepConfig(type="agent", agent_name="answerer"),
    )
    llm = _ScriptedLlm(
        model="openai/test",
        responses=[types.Content(role="model", parts=[types.Part.from_text(text="<solution>42</solution>")])],
    )
    monkeypatch.setattr(builder, "_resolve_llm", lambda *_args: llm)
    agent = builder.build(config, autonomous=False, final_answer_contract="Use <solution>answer</solution>.")
    result = await run_pipeline(agent, "Answer.", session_service=InMemorySessionService())

    assert config.final_answer_agent == "answerer"
    assert result.state[config.agents[0].output_key] == "<solution>42</solution>"
    assert result.status == "completed"


def test_ambiguous_terminal_cannot_infer_final_answer_agent():
    config = MAWConfig(
        agents=[
            MAWAgentConfig(name="a", instruction="Answer A.", output_key="a_output"),
            MAWAgentConfig(name="b", instruction="Answer B.", output_key="b_output"),
        ],
        pipeline=MAWStepConfig(
            type="parallel",
            children=[MAWStepConfig(type="agent", agent_name="a"), MAWStepConfig(type="agent", agent_name="b")],
        ),
    )
    with pytest.raises(ValueError, match="Cannot infer final_answer_agent"):
        builder.build(config, final_answer_contract="Return a final answer.")


@pytest.mark.parametrize(
    ("policy", "completed"),
    [("targeted_recovery", True), ("evidence_first", False), ("independent", False)],
)
@pytest.mark.asyncio
async def test_only_targeted_recovery_self_resolves_missing_handoff(monkeypatch, policy, completed):
    config = MAWConfig(
        agents=[
            MAWAgentConfig(
                name="producer", instruction="Return incomplete evidence.", output_key="evidence"
            ),
            MAWAgentConfig(
                name="recovery",
                instruction="Recover the missing field.",
                output_key="recovered",
                input_requirements=[ArtifactRequirement(source_key="evidence", required_fields=["claim"])],
                output_contract=ArtifactContract(required_fields=["claim"]),
                research_policy=policy,
            ),
        ],
        pipeline=MAWStepConfig(
            type="sequential",
            children=[MAWStepConfig(type="agent", agent_name="producer"), MAWStepConfig(type="agent", agent_name="recovery")],
        ),
    )
    llm = _ScriptedLlm(
        model="openai/test",
        responses=[
            types.Content(role="model", parts=[types.Part.from_text(text='{"other": "x"}')]),
            types.Content(role="model", parts=[types.Part.from_text(text='{"claim": "recovered"}')]),
        ],
    )
    monkeypatch.setattr(builder, "_resolve_llm", lambda *_args: llm)
    result = await run_pipeline(
        builder.build(config, autonomous=False), "Recover it.", session_service=InMemorySessionService()
    )

    issues = result.state["_fedotmas_execution"]["handoff_issues"]
    assert issues[0]["kind"] == "incomplete_handoff"
    assert issues[0]["resolved"] is completed
    assert result.status == ("completed" if completed else "incomplete")


@pytest.mark.asyncio
async def test_unresolved_handoff_and_malformed_artifact_remain_incomplete(monkeypatch):
    config = _contract_config().model_copy(
        update={"pipeline": MAWStepConfig(type="agent", agent_name="paper_researcher")}
    )
    llm = _ScriptedLlm(
        model="openai/test",
        responses=[types.Content(role="model", parts=[types.Part.from_text(text='{"paper_identity":"x"}')])],
    )
    monkeypatch.setattr(builder, "_resolve_llm", lambda *_args: llm)
    result = await run_pipeline(builder.build(config, autonomous=False), "Find it.", session_service=InMemorySessionService())
    assert result.status == "incomplete"
    assert result.state["_fedotmas_execution"]["handoff_issues"][0]["resolved"] is False

    malformed = _contract_config().model_copy(
        update={"pipeline": MAWStepConfig(type="agent", agent_name="paper_researcher")}
    )
    llm = _ScriptedLlm(
        model="openai/test",
        responses=[types.Content(role="model", parts=[types.Part.from_text(text="not json")])],
    )
    monkeypatch.setattr(builder, "_resolve_llm", lambda *_args: llm)
    result = await run_pipeline(builder.build(malformed, autonomous=False), "Find it.", session_service=InMemorySessionService())
    assert result.status == "incomplete"
    assert result.state["_fedotmas_execution"]["handoff_issues"][0]["kind"] == "incomplete_artifact"


@pytest.mark.asyncio
async def test_later_loop_iteration_resolves_incomplete_artifact(monkeypatch):
    config = MAWConfig(
        agents=[
            MAWAgentConfig(
                name="refiner",
                instruction="Refine the artifact.",
                output_key="artifact",
                output_contract=ArtifactContract(required_fields=["claim", "evidence"]),
            )
        ],
        pipeline=MAWStepConfig(
            type="loop",
            max_iterations=2,
            children=[MAWStepConfig(type="agent", agent_name="refiner")],
        ),
    )
    llm = _ScriptedLlm(
        model="openai/test",
        responses=[
            types.Content(role="model", parts=[types.Part.from_text(text='{"claim":"x"}')]),
            types.Content(role="model", parts=[types.Part.from_text(text='{"claim":"x","evidence":"source"}')]),
        ],
    )
    monkeypatch.setattr(builder, "_resolve_llm", lambda *_args: llm)
    result = await run_pipeline(
        builder.build(config, autonomous=False), "Refine it.", session_service=InMemorySessionService()
    )

    issue = result.state["_fedotmas_execution"]["handoff_issues"][0]
    assert issue["resolved"] is True
    assert result.status == "completed"


@pytest.mark.asyncio
async def test_later_loop_iteration_resolves_entity_continuity_mismatch(monkeypatch):
    producer, verifier = _contract_config().agents
    config = MAWConfig(
        agents=[producer, verifier],
        pipeline=MAWStepConfig(
            type="loop",
            max_iterations=2,
            children=[MAWStepConfig(type="agent", agent_name="verifier")],
        ),
    )
    llm = _ScriptedLlm(
        model="openai/test",
        responses=[
            types.Content(role="model", parts=[types.Part.from_text(text='{"paper_identity":"paper-B","decision":"supported"}')]),
            types.Content(role="model", parts=[types.Part.from_text(text='{"paper_identity":"paper-A","decision":"supported"}')]),
        ],
    )
    monkeypatch.setattr(builder, "_resolve_llm", lambda *_args: llm)
    result = await run_pipeline(
        builder.build(config, autonomous=False),
        "Verify it.",
        initial_state={"paper_evidence": json.dumps({"paper_identity": "paper-A", "equation": "E=mc^2", "evidence": "page 8"})},
        session_service=InMemorySessionService(),
    )

    issues = result.state["_fedotmas_execution"]["handoff_issues"]
    assert issues[0]["kind"] == "entity_continuity_mismatch"
    assert issues[0]["fields"] == ["paper_identity"]
    assert issues[0]["resolved"] is True
    assert result.status == "completed"


class _ScriptedLlm(BaseLlm):
    responses: list[types.Content]
    requests: list[LlmRequest] = Field(default_factory=list)

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        del stream
        self.requests.append(llm_request)
        yield LlmResponse(content=self.responses.pop(0))


@pytest.mark.asyncio
async def test_per_agent_turn_limit_stops_repeated_tool_use(monkeypatch):
    calls: list[int] = []

    def ping(value: int) -> dict[str, int]:
        """Run one bounded test action."""
        calls.append(value)
        return {"value": value}

    responses = [
        types.Content(
            role="model",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        id=f"call-{number}", name="ping", args={"value": number}
                    )
                )
            ],
        )
        for number in range(8)
    ]
    llm = _ScriptedLlm(model="openai/test", responses=responses)
    monkeypatch.setattr(builder, "_resolve_llm", lambda *_args: llm)
    config = MAWAgentConfig(
        name="looper",
        instruction="Keep using the tool for {user_query}.",
        output_key="result",
        max_llm_turns=2,
    )
    agent = builder._build_llm_agent(
        config, None, None, max_agent_llm_turns=7, autonomous=False
    )
    agent.tools = [FunctionTool(ping)]

    result = await run_pipeline(
        agent,
        "Keep working.",
        session_service=InMemorySessionService(),
    )

    assert calls == [0, 1]
    assert len(llm.requests) == 2
    assert result.status == "limited"
    assert result.state["_fedotmas_execution"]["limited_agents"]["looper"] == {
        "limit": 2,
        "turns": 2,
    }
    assert "INCOMPLETE" in result.state["result"]


@pytest.mark.asyncio
async def test_exhausted_search_budget_blocks_backend_and_bounds_retries(monkeypatch):
    backend_queries: list[str] = []

    def search(query: str) -> dict[str, str]:
        """Search the web for evidence."""
        backend_queries.append(query)
        return {"results": [{"url": "https://example.org/source"}]}

    responses = [
        types.Content(
            role="model",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        id=f"search-{number}",
                        name="search",
                        args={"query": f"query {number}"},
                    )
                )
            ],
        )
        for number in range(20)
    ]
    llm = _ScriptedLlm(model="openai/test", responses=responses)
    monkeypatch.setattr(builder, "_resolve_llm", lambda *_args: llm)
    agent = builder._build_llm_agent(
        MAWAgentConfig(
            name="researcher",
            instruction="Research {user_query}.",
            output_key="result",
            max_llm_turns=3,
        ),
        None,
        None,
        max_agent_llm_turns=8,
        autonomous=False,
    )
    agent.tools = [FunctionTool(search)]

    result = await run_pipeline(
        agent,
        "Find a source.",
        session_service=InMemorySessionService(),
        plugins=[WebSearchLimitPlugin(max_calls_per_agent=1)],
    )

    assert backend_queries == ["query 0"]
    assert len(llm.requests) == 3
    assert result.status == "limited"
    assert (
        result.state["_fedotmas_tool_budgets"]["researcher"]["search"]["status"]
        == "exhausted"
    )
    for request in llm.requests[1:]:
        advertised = [
            declaration.name
            for group in request.config.tools or []
            for declaration in group.function_declarations or []
        ]
        assert "search" not in advertised
