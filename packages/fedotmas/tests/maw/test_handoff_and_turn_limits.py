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
from fedotmas.plugins import ResearchTelemetry, WebSearchLimitPlugin
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


def test_maw_config_autofills_multiple_upstream_identity_fields():
    consumer = MAWAgentConfig(
        name="consumer",
        instruction="Use input",
        output_key="out",
        input_requirements=[
            ArtifactRequirement(source_key="up_a", identity_fields=["title"]),
            ArtifactRequirement(source_key="up_b", identity_fields=["date", "title"]),
        ],
        output_contract=ArtifactContract(required_fields=["finding"]),
    )
    upstream = [
        MAWAgentConfig(name="a", instruction="", output_key="up_a"),
        MAWAgentConfig(name="b", instruction="", output_key="up_b"),
    ]
    terminal = MAWAgentConfig(name="terminal", instruction="", output_key="answer")
    config = MAWConfig(agents=[*upstream, consumer, terminal], pipeline=MAWStepConfig(
        type="sequential", children=[MAWStepConfig(type="agent", agent_name="a"),
            MAWStepConfig(type="agent", agent_name="b"), MAWStepConfig(type="agent", agent_name="consumer"),
            MAWStepConfig(type="agent", agent_name="terminal")]
    ))
    assert config.agents[2].output_contract.identity_fields == ["date", "title"]


def test_maw_config_preserves_correct_identity_contract_and_terminal_exemption():
    existing = ArtifactContract(required_fields=["finding"], identity_fields=["id"])
    consumer = MAWAgentConfig(
        name="consumer", instruction="", output_key="out",
        input_requirements=[ArtifactRequirement(source_key="out", identity_fields=["id"])],
        output_contract=existing,
    )
    config = MAWConfig(
        agents=[consumer], pipeline=MAWStepConfig(type="agent", agent_name="consumer")
    )
    assert config.agents[0].output_contract.identity_fields == ["id"]

    terminal = MAWAgentConfig(
        name="terminal", instruction="", output_key="answer",
        input_requirements=[ArtifactRequirement(source_key="answer", identity_fields=["id"])],
    )
    terminal_config = MAWConfig(
        agents=[terminal], pipeline=MAWStepConfig(type="agent", agent_name="terminal")
    )
    assert terminal_config.agents[0].output_contract is None


def test_nonterminal_identity_consumer_is_normalized_before_validation():
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

    normalized = MAWConfig.model_validate(config.model_dump())
    assert normalized.agents[1].output_contract.identity_fields == ["paper_identity"]


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
async def test_verifier_with_missing_fields_uses_targeted_recovery_by_default():
    verifier = _contract_config().agents[1].model_copy(
        update={"research_policy": "independent"}
    )
    agent = builder._build_llm_agent(verifier, None, None, autonomous=False)
    state = {
        "paper_evidence": json.dumps(
            {"paper_identity": "paper-A", "equation": "E=mc^2"}
        )
    }
    context = _context(state)
    context._invocation_context.agent.name = "verifier"

    await agent.before_agent_callback(context)

    assert state["__fedotmas_research_policies"]["verifier"] == "targeted_recovery"


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


@pytest.mark.parametrize("policy", ["targeted_recovery", "evidence_first", "independent"])
@pytest.mark.asyncio
async def test_terminal_text_alone_does_not_resolve_missing_handoff(monkeypatch, policy):
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
    assert issue["resolved"] is False
    assert result.state["answer"] == "<solution>42</solution>"
    assert result.status == "incomplete"


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


@pytest.mark.asyncio
async def test_explicit_terminal_abstention_is_incomplete(monkeypatch):
    config = MAWConfig(
        agents=[MAWAgentConfig(name="answerer", instruction="Answer.", output_key="answer")],
        pipeline=MAWStepConfig(type="agent", agent_name="answerer"),
    )
    llm = _ScriptedLlm(
        model="openai/test",
        responses=[
            types.Content(
                role="model",
                parts=[types.Part.from_text(text="<abstain>No supported record found.</abstain>")],
            )
        ],
    )
    monkeypatch.setattr(builder, "_resolve_llm", lambda *_args: llm)

    result = await run_pipeline(
        builder.build(config, autonomous=False),
        "Find the record.",
        session_service=InMemorySessionService(),
    )

    assert result.status == "incomplete"
    assert result.state["__fedotmas_completion"] == {
        "status": "abstained",
        "reason": "No supported record found.",
        "agent": "answerer",
    }


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
async def test_contract_failure_gets_one_tools_free_format_repair(monkeypatch):
    llm = _ScriptedLlm(
        model="openai/test",
        responses=[
            types.Content(
                role="model",
                parts=[
                    types.Part.from_text(
                        text='{"claim":"supported","source":"Source already cited",'
                        '"paper_id":"paper-A"}'
                    )
                ],
            )
        ],
    )
    monkeypatch.setattr(builder, "_resolve_llm", lambda *_args: llm)
    cfg = MAWAgentConfig(
        name="producer",
        instruction="Find the claim.",
        output_key="artifact",
        output_contract=ArtifactContract(
            required_fields=["claim", "source"], identity_fields=["paper_id"]
        ),
    )
    agent = builder._build_llm_agent(cfg, None, None, autonomous=False)
    state = {
        "artifact": '{"claim":"supported","paper_id":"paper-A",'
        '"provenance":"Source already cited"}'
    }

    instruction = agent.instruction
    assert "exactly one JSON object" in instruction
    assert '"claim", "source"' in instruction
    await agent.after_agent_callback(_context(state))

    assert json.loads(state["artifact"]) == {
        "claim": "supported",
        "source": "Source already cited",
        "paper_id": "paper-A",
    }
    assert len(llm.requests) == 1
    assert not llm.requests[0].config.tools
    assert "Do not invent facts" in llm.requests[0].contents[0].parts[0].text
    assert state["_fedotmas_execution"]["contract_repairs"]["producer"] == [
        {"status": "initial_contract_failure", "missing_fields": ["source"]},
        {"status": "format_repair_succeeded"},
    ]
    assert not state["_fedotmas_execution"].get("handoff_issues")


@pytest.mark.asyncio
async def test_contract_repair_keeps_missing_semantics_incomplete(monkeypatch):
    llm = _ScriptedLlm(
        model="openai/test",
        responses=[
            types.Content(
                role="model", parts=[types.Part.from_text(text='{"claim":"x"}')]
            )
        ],
    )
    monkeypatch.setattr(builder, "_resolve_llm", lambda *_args: llm)
    cfg = MAWAgentConfig(
        name="producer",
        instruction="Find the claim.",
        output_key="artifact",
        output_contract=ArtifactContract(required_fields=["claim", "evidence"]),
    )
    agent = builder._build_llm_agent(cfg, None, None, autonomous=False)
    state = {"artifact": '{"claim":"x"}'}

    await agent.after_agent_callback(_context(state))

    assert len(llm.requests) == 1
    assert state["_fedotmas_execution"]["handoff_issues"][0]["kind"] == (
        "incomplete_artifact"
    )
    assert state["_fedotmas_execution"]["contract_repairs"]["producer"][-1] == {
        "status": "repair_missing_semantic_fields",
        "missing_fields": ["evidence"],
    }


@pytest.mark.asyncio
async def test_contract_repair_rejects_invented_required_values(monkeypatch):
    llm = _ScriptedLlm(
        model="openai/test",
        responses=[
            types.Content(
                role="model",
                parts=[
                    types.Part.from_text(
                        text='{"claim":"x","evidence":"a citation"}'
                    )
                ],
            )
        ],
    )
    monkeypatch.setattr(builder, "_resolve_llm", lambda *_args: llm)
    cfg = MAWAgentConfig(
        name="producer",
        instruction="Find the claim.",
        output_key="artifact",
        output_contract=ArtifactContract(required_fields=["claim", "evidence"]),
    )
    agent = builder._build_llm_agent(cfg, None, None, autonomous=False)
    original = '{"claim":"x","citation":"a citation"}'
    state = {"artifact": original}

    await agent.after_agent_callback(_context(state))

    assert state["artifact"] == original
    assert state["_fedotmas_execution"]["contract_repairs"]["producer"][-1] == {
        "status": "repair_missing_semantic_fields",
        "missing_fields": ["evidence"],
    }
    assert state["_fedotmas_execution"]["handoff_issues"][0]["kind"] == (
        "incomplete_artifact"
    )


@pytest.mark.asyncio
async def test_contract_repair_accepts_semantic_key_remapping(monkeypatch):
    llm = _ScriptedLlm(
        model="openai/test",
        responses=[
            types.Content(
                role="model",
                parts=[types.Part.from_text(text='{"solutions":[[7,9]]}')],
            )
        ],
    )
    monkeypatch.setattr(builder, "_resolve_llm", lambda *_args: llm)
    cfg = MAWAgentConfig(
        name="producer",
        instruction="Find the valid pairs.",
        output_key="artifact",
        output_contract=ArtifactContract(required_fields=["solutions"]),
    )
    agent = builder._build_llm_agent(cfg, None, None, autonomous=False)
    state = {"artifact": '{"valid_pairs":[[7,9]]}'}

    await agent.after_agent_callback(_context(state))

    assert json.loads(state["artifact"]) == {"solutions": [[7, 9]]}
    assert state["_fedotmas_execution"]["contract_repairs"]["producer"][-1] == {
        "status": "format_repair_succeeded"
    }


@pytest.mark.asyncio
async def test_contract_repair_copies_unambiguous_nested_value_and_preserves_valid_fields(monkeypatch):
    llm = _ScriptedLlm(
        model="openai/test",
        responses=[types.Content(role="model", parts=[types.Part.from_text(
            text='{"answer":"A","standard_name":"X","assessments":[{"standard_name":"X","status":"superseded"}]}'
        )])],
    )
    monkeypatch.setattr(builder, "_resolve_llm", lambda *_args: llm)
    cfg = MAWAgentConfig(
        name="producer", instruction="", output_key="artifact",
        output_contract=ArtifactContract(required_fields=["answer", "standard_name", "assessments"]),
    )
    agent = builder._build_llm_agent(cfg, None, None, autonomous=False)
    state = {"artifact": '{"answer":"A","assessments":[{"standard_name":"X","status":"superseded"}]}' }
    await agent.after_agent_callback(_context(state))
    assert json.loads(state["artifact"]) == {
        "answer": "A", "standard_name": "X",
        "assessments": [{"standard_name": "X", "status": "superseded"}],
    }


@pytest.mark.asyncio
async def test_contract_repair_rejects_ambiguous_nested_mapping_and_changed_valid_value(monkeypatch):
    llm = _ScriptedLlm(model="openai/test", responses=[types.Content(role="model", parts=[
        types.Part.from_text(text='{"answer":"B","source":"src","standard_name":"X"}')
    ])])
    monkeypatch.setattr(builder, "_resolve_llm", lambda *_args: llm)
    cfg = MAWAgentConfig(name="producer", instruction="", output_key="artifact",
        output_contract=ArtifactContract(required_fields=["answer", "source", "standard_name"]))
    agent = builder._build_llm_agent(cfg, None, None, autonomous=False)
    original = '{"answer":"A","solution":"B","provenance":"src","assessments":[{"standard_name":"X"},{"standard_name":"Y"}]}'
    state = {"artifact": original}
    await agent.after_agent_callback(_context(state))
    assert state["artifact"] == original


def test_contract_repair_rejects_ambiguous_nested_value_mapping():
    cfg = MAWAgentConfig(name="producer", instruction="", output_key="artifact",
        output_contract=ArtifactContract(required_fields=["standard_name"]))
    assert builder._repair_values_supported(
        '{"assessments":[{"standard_name":"X"},{"standard_name":"Y"}]}',
        '{"standard_name":"X"}', cfg, {},
    ) == ["standard_name"]


@pytest.mark.asyncio
async def test_builder_discovery_inspection_gate_transitions_through_model_callbacks():
    search = FunctionTool(func=lambda query: {"results": []})
    search.name = "searxng_search"
    search.description = "Search the web"
    inspect = FunctionTool(func=lambda url: {"content": ""})
    inspect.name = "markdown"
    inspect.description = "Inspect a known page"
    cfg = MAWAgentConfig(name="researcher", instruction="Research", output_key="out")
    agent = builder._build_llm_agent(cfg, [search, inspect], None, autonomous=False)
    state: dict = {}
    context = _context(state)
    context._invocation_context.agent.name = "researcher"
    telemetry = ResearchTelemetry()
    await agent.before_agent_callback(context)
    request = _research_request(search, inspect)
    await agent.before_model_callback(context, request)
    assert search.name in {
        declaration.name
        for group in request.config.tools or []
        for declaration in group.function_declarations or []
    }

    args = {"query": "specific source"}
    assert await agent.before_tool_callback(search, args, context) is None
    await telemetry.before_tool_callback(tool=search, tool_args=args, tool_context=context)
    await telemetry.after_tool_callback(
        tool=search,
        tool_args=args,
        tool_context=context,
        result={
            "results": [
                {
                    "url": "https://example.org/source",
                    "title": "Primary source",
                    "snippet": "The exact requested record.",
                }
            ]
        },
    )

    request = _research_request(search, inspect)
    await agent.before_model_callback(context, request)
    exposed = {
        declaration.name
        for group in request.config.tools or []
        for declaration in group.function_declarations or []
    }
    assert search.name not in exposed
    assert inspect.name in exposed
    assert "Primary source" in request.config.system_instruction

    blocked = await agent.before_tool_callback(
        search, {"query": "another broad query"}, context
    )
    assert blocked["error_code"] == "INSPECT_CANDIDATES_FIRST"

    inspect_args = {"url": "https://example.org/source"}
    await telemetry.before_tool_callback(
        tool=inspect, tool_args=inspect_args, tool_context=context
    )
    await telemetry.after_tool_callback(
        tool=inspect,
        tool_args=inspect_args,
        tool_context=context,
        result={"content": "The record reports value 17."},
    )
    request = _research_request(search, inspect)
    await agent.before_model_callback(context, request)
    assert search.name in {
        declaration.name
        for group in request.config.tools or []
        for declaration in group.function_declarations or []
    }
    ledger = state["__fedotmas_research_candidates"]["researcher"]
    assert ledger[0]["inspected"] is True
    assert telemetry.snapshot()["researcher"]["candidate_urls_inspected"] == 1


def _research_request(*tools):
    return LlmRequest(
        tools_dict={tool.name: tool for tool in tools},
        config=types.GenerateContentConfig(
            tools=[
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(
                            name=tool.name, description=tool.description
                        )
                        for tool in tools
                    ]
                )
            ]
        ),
    )


@pytest.mark.asyncio
async def test_discovery_only_source_finder_does_not_wait_for_inspection():
    telemetry = ResearchTelemetry()
    search = FunctionTool(func=lambda query: {"results": []})
    search.name = "searxng_search"
    inspect = FunctionTool(func=lambda url: {"content": ""})
    inspect.name = "markdown"
    agent = builder._build_llm_agent(
        MAWAgentConfig(
            name="source_finder",
            instruction="Find and select candidate sources.",
            output_key="sources",
            research_mode="discovery_only",
        ),
        [search, inspect],
        None,
        autonomous=False,
    )
    state: dict = {}
    context = _context(state)
    context._invocation_context.agent.name = "source_finder"
    await agent.before_agent_callback(context)
    request = _research_request(search, inspect)
    await agent.before_model_callback(context, request)
    exposed = {
        declaration.name
        for group in request.config.tools or []
        for declaration in group.function_declarations or []
    }
    assert search.name in exposed
    assert inspect.name in exposed
    args = {"query": "candidate sources"}
    assert await telemetry.before_tool_callback(
        tool=search, tool_args=args, tool_context=context
    ) is None
    await telemetry.after_tool_callback(
        tool=search,
        tool_args=args,
        tool_context=context,
        result={"results": [{"url": "https://example.org/source", "title": "A source"}]},
    )

    assert "source_finder" not in state.get("__fedotmas_research_gate", {})
    assert await telemetry.before_tool_callback(
        tool=search,
        tool_args={"query": "another source query"},
        tool_context=context,
    ) is None


@pytest.mark.asyncio
async def test_source_finder_can_validate_video_identity_but_not_read_transcript():
    search = FunctionTool(func=lambda query: {"results": []})
    search.name = "searxng_search"
    info = FunctionTool(func=lambda video_id: {"title": "Known video"})
    info.name = "get_video_info"
    transcript = FunctionTool(func=lambda video_id: {"text": "long transcript"})
    transcript.name = "get_transcript"
    agent = builder._build_llm_agent(MAWAgentConfig(name="finder", instruction="Find source", output_key="out", research_mode="discovery_only"), [search, info, transcript], None, autonomous=False)
    context = _context({"__fedotmas_research_candidates": {"finder": [
        {"url": "https://www.youtube.com/watch?v=abc", "title": "Known video"}
    ]}})
    context._invocation_context.agent.name = "finder"
    await agent.before_agent_callback(context)
    request = _research_request(search, info, transcript)
    await agent.before_model_callback(context, request)
    visible = {d.name for g in request.config.tools or [] for d in g.function_declarations or []}
    assert info.name in visible
    assert transcript.name not in visible
    assert await agent.before_tool_callback(info, {"video_id": "abc"}, context) is None
    assert (await agent.before_tool_callback(transcript, {"video_id": "abc"}, context))["error_code"] == "DISCOVERY_ONLY_INSPECTION_DISABLED"


@pytest.mark.asyncio
async def test_discovery_only_can_validate_two_known_markdown_candidates_only():
    search = FunctionTool(func=lambda query: {"results": []})
    search.name = "search"
    markdown = FunctionTool(func=lambda url: {"content": "candidate page"})
    markdown.name = "markdown"
    interactive = [FunctionTool(func=dict) for _ in range(5)]
    for tool, name in zip(
        interactive, ["click", "scroll", "semantic_tree", "findElement", "nodeDetails"]
    ):
        tool.name = name
    state = {"__fedotmas_research_candidates": {"finder": [
        {"url": "https://example.org/a"}, {"url": "https://example.org/b"}
    ]}}
    agent = builder._build_llm_agent(
        MAWAgentConfig(name="finder", instruction="Find source", output_key="out", research_mode="discovery_only"),
        [search, markdown, *interactive], None, autonomous=False,
    )
    context = _context(state)
    context._invocation_context.agent.name = "finder"
    await agent.before_agent_callback(context)
    request = _research_request(search, markdown)
    await agent.before_model_callback(context, request)
    visible = {d.name for g in request.config.tools or [] for d in g.function_declarations or []}
    assert markdown.name in visible
    assert not {tool.name for tool in interactive} & visible
    assert await agent.before_tool_callback(markdown, {"url": "https://unknown.example"}, context)
    assert await agent.before_tool_callback(markdown, {"url": "https://example.org/a"}, context) is None
    assert await agent.before_tool_callback(markdown, {"url": "https://example.org/b"}, context) is None
    assert (await agent.before_tool_callback(markdown, {"url": "https://example.org/a"}, context))["error_code"] == "DISCOVERY_VALIDATION_LIMIT"


@pytest.mark.asyncio
async def test_targeted_recovery_is_one_identity_scoped_search_and_browser_is_anchored():
    search = FunctionTool(func=lambda query: {"results": []})
    search.name = "searxng_search"
    search.description = "Search the web for a known source"
    browser = FunctionTool(func=lambda task: {"status": "ok"})
    browser.name = "complete_browser_task"
    agent = builder._build_llm_agent(MAWAgentConfig(name="extractor", instruction="Inspect source", output_key="out", research_mode="inspection_only", research_policy="targeted_recovery"), [search, browser], None, autonomous=False)
    state = {"__fedotmas_research_candidates": {"extractor": [{"url": "https://example.org/paper", "title": "Known Paper DOI 10.1234/known", "snippet": "", "inspected": False}]}}
    context = _context(state)
    context._invocation_context.agent.name = "extractor"
    await agent.before_agent_callback(context)
    request = _research_request(search, browser)
    await agent.before_model_callback(context, request)
    open_search = await agent.before_tool_callback(browser, {"task": "search the web for similar papers"}, context)
    assert open_search["error_code"] == "BROWSER_DISCOVERY_POLICY"
    assert await agent.before_tool_callback(search, {"query": "find another copy of Known Paper DOI 10.1234/known"}, context) is None
    request = _research_request(search, browser)
    await agent.before_model_callback(context, request)
    assert search.name not in {d.name for g in request.config.tools or [] for d in g.function_declarations or []}


@pytest.mark.asyncio
async def test_forced_convergence_blocks_controller_and_open_browser_loops():
    controller = FunctionTool(func=lambda: {"action": "search again"})
    controller.name = "get_next_action"
    browser = FunctionTool(func=lambda task: {"status": "ok"})
    browser.name = "complete_browser_task"
    inspect = FunctionTool(func=lambda url: {"content": ""})
    inspect.name = "markdown"
    agent = builder._build_llm_agent(MAWAgentConfig(name="researcher", instruction="Research", output_key="out"), [controller, browser, inspect], None, autonomous=False)
    state: dict = {}
    context = _context(state)
    context._invocation_context.agent.name = "researcher"
    await agent.before_agent_callback(context)
    for _ in range(3):
        await agent.before_model_callback(context, _research_request(controller, browser, inspect))
        await agent.after_tool_callback(
            inspect, {"url": f"https://example.org/{_}"}, context, {"content": ""}
        )
    assert state["__fedotmas_research_turns"]["researcher"]["force_converge"] is True
    assert (await agent.before_tool_callback(controller, {}, context))["error_code"] == "RESEARCH_CONVERGENCE_REQUIRED"
    assert (await agent.before_tool_callback(browser, {"task": "search the web"}, context))["error_code"] in {"BROWSER_DISCOVERY_POLICY", "RESEARCH_CONVERGENCE_REQUIRED"}


@pytest.mark.asyncio
async def test_controller_errors_before_evidence_do_not_force_research_convergence():
    update = FunctionTool(func=lambda **kwargs: {"research_state": {"version": 1}})
    update.name = "update_research_state"
    controller = FunctionTool(func=lambda research_state: {"action": "continue_search"})
    controller.name = "get_next_action"
    search = FunctionTool(func=lambda query: {"results": [{"url": "https://example.org"}]})
    search.name = "search"
    search.description = "Search the web"
    agent = builder._build_llm_agent(
        MAWAgentConfig(name="researcher", instruction="Research", output_key="out"),
        [update, controller, search], None, autonomous=False,
    )
    state: dict = {}
    context = _context(state)
    context._invocation_context.agent.name = "researcher"
    await agent.before_agent_callback(context)

    for _ in range(3):
        request = _research_request(update, controller, search)
        await agent.before_model_callback(context, request)
        visible = {d.name for group in request.config.tools or [] for d in group.function_declarations or []}
        assert controller.name not in visible
    for _ in range(2):
        blocked = await agent.before_tool_callback(controller, {"research_state": {}}, context)
        assert blocked["error_code"] == "RESEARCH_CONTROLLER_STATE_REQUIRED"
        await agent.before_model_callback(context, _research_request(update, controller, search))
    assert state[builder.RESEARCH_TURN_STATE_KEY]["researcher"]["force_converge"] is False
    assert await agent.before_tool_callback(search, {"query": "PubChem"}, context) is None

    await agent.after_tool_callback(
        update, {}, context, {"research_state": {"version": 1}}
    )
    assert await agent.before_tool_callback(controller, {"research_state": {}}, context) is None
    await agent.after_tool_callback(search, {"query": "PubChem"}, context, {"results": []})
    assert state[builder.RESEARCH_EVIDENCE_ACTION_STATE_KEY]["researcher"] == 1
    for _ in range(3):
        await agent.before_model_callback(context, _research_request(update, controller, search))
        await agent.after_tool_callback(
            search, {"query": "another search"}, context, {"results": []}
        )
    assert state[builder.RESEARCH_TURN_STATE_KEY]["researcher"]["force_converge"] is True, state[builder.RESEARCH_PROGRESS_STATE_KEY]


@pytest.mark.asyncio
async def test_discovery_only_source_finder_gets_second_productive_wave_before_handoff():
    telemetry = ResearchTelemetry()
    search = FunctionTool(func=lambda query: {"results": []})
    search.name = "searxng_search"
    search.description = "Search the web"
    inspect = FunctionTool(func=lambda url: {"content": ""})
    inspect.name = "markdown"
    inspect.description = "Inspect a known URL"
    agent = builder._build_llm_agent(
        MAWAgentConfig(
            name="source_finder",
            instruction="Find primary source candidates.",
            output_key="sources",
            research_mode="discovery_only",
        ),
        [search, inspect],
        None,
        autonomous=False,
    )
    state: dict = {}
    context = _context(state)
    context._invocation_context.agent.name = "source_finder"
    await agent.before_agent_callback(context)
    request = _research_request(search, inspect)
    await agent.before_model_callback(context, request)
    args = {"query": "exact record"}
    assert await agent.before_tool_callback(search, args, context) is None
    await telemetry.before_tool_callback(
        tool=search, tool_args=args, tool_context=context
    )
    await telemetry.after_tool_callback(
        tool=search,
        tool_args=args,
        tool_context=context,
        result={
            "results": [
                {
                    "url": f"https://example.org/source-{index}",
                    "title": f"Source {index}",
                    "snippet": "An exact record match.",
                }
                for index in range(3)
            ]
        },
    )

    request = _research_request(search, inspect)
    await agent.before_model_callback(context, request)
    exposed = {
        declaration.name
        for group in request.config.tools or []
        for declaration in group.function_declarations or []
    }
    assert search.name in exposed
    assert inspect.name in exposed
    assert "Source 0" in request.config.system_instruction
    assert await agent.before_tool_callback(search, {"query": "exact DOI source"}, context) is None
    await telemetry.before_tool_callback(tool=search, tool_args={"query": "exact DOI source"}, tool_context=context)
    await telemetry.after_tool_callback(tool=search, tool_args={"query": "exact DOI source"}, tool_context=context, result={"results": [{"url": "https://example.org/exact", "title": "Exact source"}]})
    assert any(item["url"] == "https://example.org/exact" for item in state["__fedotmas_research_candidates"]["source_finder"])
    state["__fedotmas_research_progress"]["source_finder"]["productive_discovery_waves"] = 3
    request = _research_request(search, inspect)
    await agent.before_model_callback(context, request)
    exposed = {declaration.name for group in request.config.tools or [] for declaration in group.function_declarations or []}
    assert search.name not in exposed
    assert "Exact source" in request.config.system_instruction
    blocked_inspection = await agent.before_tool_callback(
        inspect, {"url": "https://example.org/source-0"}, context
    )
    assert blocked_inspection is None
    assert (await agent.before_tool_callback(
        inspect, {"url": "https://not-discovered.example"}, context
    ))["error_code"] == "DISCOVERY_VALIDATION_REQUIRES_CANDIDATE"


@pytest.mark.asyncio
async def test_same_turn_broad_discovery_fanout_is_capped_but_inspection_is_allowed():
    search = FunctionTool(func=lambda query: {"results": []})
    search.name = "searxng_search"
    search.description = "Search the web"
    inspect = FunctionTool(func=lambda url: {"content": ""})
    inspect.name = "markdown"
    inspect.description = "Inspect a known URL"
    agent = builder._build_llm_agent(
        MAWAgentConfig(name="researcher", instruction="Research", output_key="out"),
        [search, inspect],
        None,
        autonomous=False,
    )
    state: dict = {}
    context = _context(state)
    context._invocation_context.agent.name = "researcher"
    await agent.before_agent_callback(context)
    request = _research_request(search, inspect)
    await agent.before_model_callback(context, request)

    assert await agent.before_tool_callback(search, {"query": "first"}, context) is None
    blocked = await agent.before_tool_callback(search, {"query": "second"}, context)
    assert blocked["error_code"] == "DISCOVERY_FANOUT_LIMIT"
    assert await agent.before_tool_callback(
        inspect, {"url": "https://example.org/known"}, context
    ) is None
    assert "Per-turn research budget" in request.config.system_instruction

    await agent.before_model_callback(context, _research_request(search, inspect))
    assert await agent.before_tool_callback(search, {"query": "next turn"}, context) is None


@pytest.mark.asyncio
async def test_model_search_fanout_executes_only_one_call_per_response(monkeypatch):
    executed: list[str] = []

    def search(query: str) -> dict:
        """Search broadly for candidate sources."""
        executed.append(query)
        return {"results": [{"url": f"https://example.org/{query}"}]}

    llm = _ScriptedLlm(
        model="openai/test",
        responses=[
            types.Content(
                role="model",
                parts=[
                    types.Part(
                        function_call=types.FunctionCall(
                            id="search-1", name="search", args={"query": "first"}
                        )
                    ),
                    types.Part(
                        function_call=types.FunctionCall(
                            id="search-2", name="search", args={"query": "second"}
                        )
                    ),
                ],
            ),
            types.Content(role="model", parts=[types.Part.from_text(text="done")]),
        ],
    )
    monkeypatch.setattr(builder, "_resolve_llm", lambda *_args: llm)
    agent = builder._build_llm_agent(
        MAWAgentConfig(
            name="researcher",
            instruction="Research.",
            output_key="answer",
            research_mode="discovery_only",
        ),
        None,
        None,
        autonomous=False,
    )
    agent.tools = [FunctionTool(search)]

    telemetry = ResearchTelemetry()
    result = await run_pipeline(
        agent,
        "Find one useful source.",
        session_service=InMemorySessionService(),
        plugins=[telemetry],
    )

    assert executed == ["first"]
    turns = result.state["_fedotmas_execution"]["turn_observability"]["researcher"]
    turn = turns[0]
    assert [call["tool"] for call in turn["tool_calls_selected"]] == ["search"]
    assert [call["reason"] for call in turn["calls_blocked"]] == [
        "DISCOVERY_FANOUT_LIMIT"
    ]
    assert turn["visible_tool_declarations"][0]["name"] == "search"
    assert "Search broadly for candidate sources" in turn["visible_tool_declarations"][0]["description"]
    assert "candidate_evidence" not in turns[1]["semantic_progress_events"]


@pytest.mark.asyncio
async def test_candidate_inspection_progress_is_bounded_during_agent_lifecycle():
    search = FunctionTool(func=lambda query: {"results": []})
    search.name = "searxng_search"
    search.description = "Search the web"
    inspect = FunctionTool(func=lambda url: {"content": "candidate source text"})
    inspect.name = "markdown"
    inspect.description = "Read a known candidate page"
    agent = builder._build_llm_agent(
        MAWAgentConfig(name="researcher", instruction="Research", output_key="out"),
        [search, inspect],
        None,
        autonomous=False,
    )
    state: dict = {}
    context = _context(state)
    context._invocation_context.agent.name = "researcher"
    telemetry = ResearchTelemetry()
    tool_context = MagicMock()
    tool_context._invocation_context.agent.name = "researcher"
    tool_context.state = state
    await agent.before_agent_callback(context)

    async def next_turn():
        request = _research_request(search, inspect)
        await agent.before_model_callback(context, request)
        return request

    async def inspect_candidate(url: str):
        args = {"url": url}
        assert await agent.before_tool_callback(inspect, args, context) is None
        await telemetry.before_tool_callback(
            tool=inspect, tool_args=args, tool_context=tool_context
        )
        await telemetry.after_tool_callback(
            tool=inspect,
            tool_args=args,
            tool_context=tool_context,
            result={"content": "The candidate page identifies the requested source."},
        )
        await agent.after_tool_callback(
            inspect,
            args,
            context,
            {"content": "The candidate page identifies the requested source."},
        )

    request = await next_turn()
    assert search.name in {
        declaration.name
        for group in request.config.tools or []
        for declaration in group.function_declarations or []
    }
    search_args = {"query": "requested source"}
    assert await agent.before_tool_callback(search, search_args, context) is None
    await telemetry.before_tool_callback(
        tool=search, tool_args=search_args, tool_context=tool_context
    )
    candidates = [
        {
            "url": f"https://example.org/source-{index}",
            "title": f"Source {index}",
            "snippet": f"Evidence identifying source {index}.",
        }
        for index in range(3)
    ]
    await telemetry.after_tool_callback(
        tool=search,
        tool_args=search_args,
        tool_context=tool_context,
        result={"results": candidates},
    )

    for index in range(3):
        request = await next_turn()
        assert inspect.name in {
            declaration.name
            for group in request.config.tools or []
            for declaration in group.function_declarations or []
        }
        await inspect_candidate(candidates[index]["url"])

    request = await next_turn()
    visible = {
        declaration.name
        for group in request.config.tools or []
        for declaration in group.function_declarations or []
    }
    assert search.name in visible
    assert state["__fedotmas_research_progress"]["researcher"][
        "candidate_inspection_progress_count"
    ] == 3
    assert request.config.tools

    await inspect_candidate(candidates[0]["url"])
    await next_turn()
    await inspect_candidate(candidates[0]["url"])
    request = await next_turn()
    assert search.name not in {
        declaration.name
        for group in request.config.tools or []
        for declaration in group.function_declarations or []
    }
    assert state["__fedotmas_research_turns"]["researcher"]["force_converge"] is True
    assert (
        await agent.before_tool_callback(
            inspect, {"url": candidates[0]["url"]}, context
        )
    )["error_code"] == "RESEARCH_CONVERGENCE_REQUIRED"


@pytest.mark.asyncio
async def test_repeated_no_progress_turns_hide_broad_discovery_and_are_recorded():
    search = FunctionTool(func=lambda query: {"results": []})
    search.name = "searxng_search"
    search.description = "Search the web"
    inspect = FunctionTool(func=lambda url: {"content": ""})
    inspect.name = "markdown"
    inspect.description = "Inspect a known URL"
    agent = builder._build_llm_agent(
        MAWAgentConfig(name="researcher", instruction="Research", output_key="out"),
        [search, inspect],
        None,
        autonomous=False,
    )
    state: dict = {}
    state[builder.RESEARCH_EVIDENCE_ACTION_STATE_KEY] = {"researcher": 1}
    context = _context(state)
    context._invocation_context.agent.name = "researcher"
    telemetry = ResearchTelemetry()
    inspect_context = MagicMock()
    inspect_context._invocation_context.agent.name = "researcher"
    inspect_context.state = state
    await agent.before_agent_callback(context)
    for _ in range(3):
        request = _research_request(search, inspect)
        await agent.before_model_callback(context, request)
        await telemetry.before_tool_callback(
            tool=inspect, tool_args={"url": f"https://example.org/{_ + 1}"},
            tool_context=inspect_context,
        )
        await telemetry.after_tool_callback(
            tool=inspect, tool_args={"url": f"https://example.org/{_ + 1}"},
            tool_context=inspect_context, result={"content": "another page"},
        )
        await agent.after_tool_callback(
            inspect,
            {"url": f"https://example.org/{_ + 1}"},
            inspect_context,
            {"content": "another page"},
        )

    trace = state["_fedotmas_execution"]["turn_observability"]["researcher"]
    assert trace[-1]["no_progress_turns"] == 2
    assert state["_fedotmas_execution"]["research_progress"]["researcher"]["no_progress_turns"] == 2
    assert search.name not in {
        declaration.name
        for group in request.config.tools or []
        for declaration in group.function_declarations or []
    }
    assert "repeated turns without semantic progress" in {
        item["reason"] for item in trace[-1]["removed_tools"]
    }
    assert (await agent.before_tool_callback(
        inspect, {"url": "https://example.org/further-page"}, context
    ))["error_code"] == "RESEARCH_CONVERGENCE_REQUIRED"


@pytest.mark.asyncio
async def test_invalid_calls_do_not_spend_evidence_stagnation_turns():
    search = FunctionTool(func=lambda query: {"results": [{"url": "https://example.org/a"}]})
    search.name = "searxng_search"
    inspect = FunctionTool(func=lambda url: {"content": ""})
    inspect.name = "markdown"
    agent = builder._build_llm_agent(
        MAWAgentConfig(name="researcher", instruction="Research", output_key="out"),
        [search, inspect],
        None,
        autonomous=False,
    )
    state: dict = {}
    context = _context(state)
    context._invocation_context.agent.name = "researcher"
    tool_context = MagicMock()
    tool_context._invocation_context.agent.name = "researcher"
    tool_context.state = state
    await agent.before_agent_callback(context)

    # First turn: a successful discovery and inspection, both without progress.
    await agent.before_model_callback(context, _research_request(search, inspect))
    await agent.after_tool_callback(
        search, {"query": "topic"}, tool_context,
        {"results": [{"url": "https://example.org/a"}]},
    )
    await agent.after_tool_callback(
        inspect, {"url": "https://example.org/a"}, tool_context, {"content": ""}
    )

    # The next turn records one stagnant evidence step. Invalid calls don't add
    # evidence attempts, so a following legitimate inspection remains allowed.
    await agent.before_model_callback(context, _research_request(search, inspect))
    assert state[builder.RESEARCH_TURN_STATE_KEY]["researcher"]["force_converge"] is False
    for args in ({"url": ""}, {"url": ""}):
        await agent.after_tool_callback(
            inspect, args, tool_context,
            {"isError": True, "error_code": "INVALID_TOOL_INPUT"},
        )
    await agent.before_model_callback(context, _research_request(search, inspect))
    assert state[builder.RESEARCH_TURN_STATE_KEY]["researcher"]["force_converge"] is False
    assert await agent.before_tool_callback(
        inspect, {"url": "https://example.org/b"}, context
    ) is None
    await agent.after_tool_callback(
        inspect, {"url": "https://example.org/b"}, tool_context, {"content": ""}
    )

    # Three genuine no-progress inspections eventually force convergence.
    await agent.before_model_callback(context, _research_request(search, inspect))
    assert state[builder.RESEARCH_TURN_STATE_KEY]["researcher"]["force_converge"] is True


@pytest.mark.asyncio
async def test_resolved_required_output_field_resets_no_progress():
    search = FunctionTool(func=lambda query: {"results": []})
    search.name = "search"
    cfg = MAWAgentConfig(
        name="researcher", instruction="Research", output_key="artifact",
        output_contract=ArtifactContract(required_fields=["finding"]),
    )
    agent = builder._build_llm_agent(cfg, [search], None, autonomous=False)
    state = {
        "artifact": '{"finding":"supported by evidence"}',
        "__fedotmas_execution": {"agent_llm_turns": {"researcher": 1}},
        "__fedotmas_research_progress": {
            "researcher": {"version": 0, "last_turn_version": 0, "no_progress_turns": 2}
        },
    }
    await agent.after_agent_callback(_context(state))
    request = _research_request(search)
    await agent.before_model_callback(_context(state), request)

    progress = state["__fedotmas_research_progress"]["researcher"]
    assert progress["version"] == 1
    assert progress["last_turn_version"] == 1
    assert state["__fedotmas_research_turns"]["researcher"]["force_converge"] is False
    assert any(
        item.startswith("handoff_field:finding")
        for item in state["__fedotmas_research_progress"]["researcher"]["progress_events"]
    )


@pytest.mark.asyncio
async def test_complete_evidence_verifier_cannot_broad_search():
    search = FunctionTool(func=lambda query: {"results": []})
    search.name = "searxng_search"
    search.description = "Search the web"
    verifier = builder._build_llm_agent(
        MAWAgentConfig(
            name="verifier",
            instruction="Verify the claim using the supplied evidence.",
            output_key="verification",
            input_requirements=[
                ArtifactRequirement(
                    source_key="evidence", required_fields=["claim", "source"]
                )
            ],
        ),
        [search],
        None,
        autonomous=False,
    )
    state = {"evidence": '{"claim":"value is 17","source":"official record"}'}
    context = _context(state)
    context._invocation_context.agent.name = "verifier"
    await verifier.before_agent_callback(context)
    request = _research_request(search)
    await verifier.before_model_callback(context, request)

    assert search.name not in {
        declaration.name
        for group in request.config.tools or []
        for declaration in group.function_declarations or []
    }
    blocked = await verifier.before_tool_callback(search, {"query": "broad query"}, context)
    assert blocked["error_code"] == "EVIDENCE_FIRST_SEARCH_DISABLED"


@pytest.mark.asyncio
async def test_nested_repeated_identity_contract_preserves_each_orcid():
    people = [
        {"name": "Ada Example", "orcid": "0000-0001-1111-1111"},
        {"name": "Lin Example", "orcid": "0000-0002-2222-2222"},
    ]
    contract = ArtifactContract(
        required_fields=["people"], identity_fields=["people[].orcid"]
    )
    cfg = MAWAgentConfig(
        name="producer",
        instruction="Preserve the identity of each person.",
        output_key="artifact",
        output_contract=contract,
    )
    agent = builder._build_llm_agent(cfg, None, None, autonomous=False)
    state = {"artifact": json.dumps({"people": people})}

    await agent.after_agent_callback(_context(state))

    assert json.loads(state["artifact"]) == {"people": people}
    assert "orcid" not in json.loads(state["artifact"])
    assert not state.get("_fedotmas_execution", {}).get("handoff_issues")


@pytest.mark.asyncio
async def test_contract_repair_rejects_identity_copied_from_unrelated_field(monkeypatch):
    llm = _ScriptedLlm(
        model="openai/test",
        responses=[
            types.Content(
                role="model",
                parts=[types.Part.from_text(text='{"paper_id":"paper-B"}')],
            )
        ],
    )
    monkeypatch.setattr(builder, "_resolve_llm", lambda *_args: llm)
    cfg = MAWAgentConfig(
        name="producer",
        instruction="Preserve the paper identity.",
        output_key="artifact",
        input_requirements=[
            ArtifactRequirement(source_key="upstream", identity_fields=["paper_id"])
        ],
        output_contract=ArtifactContract(
            required_fields=["claim"], identity_fields=["paper_id"]
        ),
    )
    agent = builder._build_llm_agent(cfg, None, None, autonomous=False)
    original = '{"claim":"supported","note":"paper-A"}'
    state = {
        "artifact": original,
        "upstream": '{"paper_id":"paper-A"}',
    }

    await agent.after_agent_callback(_context(state))

    assert state["artifact"] == original
    assert state["_fedotmas_execution"]["handoff_issues"][0]["kind"] == (
        "incomplete_artifact"
    )
    assert "paper_id" in state["_fedotmas_execution"]["contract_repairs"][
        "producer"
    ][-1]["missing_fields"]


@pytest.mark.asyncio
async def test_repaired_artifact_is_validated_and_consumed_downstream(monkeypatch):
    producer = MAWAgentConfig(
        name="producer",
        instruction="Find valid pairs.",
        output_key="artifact",
        output_contract=ArtifactContract(required_fields=["solutions"]),
    )
    consumer = MAWAgentConfig(
        name="consumer",
        instruction="Use this artifact: {artifact}",
        output_key="consumed",
        input_requirements=[
            ArtifactRequirement(source_key="artifact", required_fields=["solutions"])
        ],
    )
    config = MAWConfig(
        agents=[producer, consumer],
        pipeline=MAWStepConfig(
            type="sequential",
            children=[
                MAWStepConfig(type="agent", agent_name="producer"),
                MAWStepConfig(type="agent", agent_name="consumer"),
            ],
        ),
    )
    llm = _ScriptedLlm(
        model="openai/test",
        responses=[
            types.Content(
                role="model",
                parts=[types.Part.from_text(text='{"valid_pairs":[[7,9]]}')],
            ),
            types.Content(
                role="model",
                parts=[types.Part.from_text(text='{"solutions":[[7,9]]}')],
            ),
            types.Content(
                role="model",
                parts=[types.Part.from_text(text='{"consumed":true}')],
            ),
        ],
    )
    monkeypatch.setattr(builder, "_resolve_llm", lambda *_args: llm)

    result = await run_pipeline(
        builder.build(config, autonomous=False),
        "Find pairs.",
        session_service=InMemorySessionService(),
    )

    assert result.status == "completed"
    assert json.loads(result.state["artifact"]) == {"solutions": [[7, 9]]}
    assert not result.state["_fedotmas_execution"].get("handoff_issues")
    assert len(llm.requests) == 3
    consumer_request = llm.requests[-1]
    assert "solutions" in str(consumer_request.config.system_instruction)


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


@pytest.mark.asyncio
async def test_sequential_extractor_uses_upstream_source_for_browser_and_recovery(monkeypatch):
    finder_llm = _ScriptedLlm(
        model="openai/finder",
        responses=[types.Content(role="model", parts=[types.Part.from_text(text=json.dumps({"sources": [{"title": "Known source", "doi": "10.1234/abc"}]}))])],
    )
    extractor_llm = _ScriptedLlm(
        model="openai/extractor",
        responses=[
            types.Content(role="model", parts=[types.Part(function_call=types.FunctionCall(id="browser-1", name="complete_browser_task", args={"task": "Open https://doi.org/10.1234/abc"}))]),
            types.Content(role="model", parts=[types.Part(function_call=types.FunctionCall(id="search-1", name="searxng_search", args={"query": "find another copy of Known source DOI 10.1234/abc"}))]),
            types.Content(role="model", parts=[types.Part.from_text(text='{"evidence":"recovered"}')]),
        ],
    )
    llms = {"openai/finder": finder_llm, "openai/extractor": extractor_llm}
    monkeypatch.setattr(builder, "_resolve_llm", lambda model, *_args: llms[model])
    config = MAWConfig(
        agents=[
            MAWAgentConfig(name="source_finder", instruction="Find the source", output_key="source_handoff", model="openai/finder"),
            MAWAgentConfig(name="extractor", instruction="Inspect and extract evidence", output_key="evidence", model="openai/extractor", research_mode="inspection_only", research_policy="targeted_recovery", input_requirements=[ArtifactRequirement(source_key="source_handoff", required_fields=["sources"], identity_fields=["sources[].doi"])]),
        ],
        pipeline=MAWStepConfig(type="sequential", children=[MAWStepConfig(type="agent", agent_name="source_finder"), MAWStepConfig(type="agent", agent_name="extractor")]),
    )
    root = builder.build(config, autonomous=False)
    opened: list[str] = []
    searches: list[str] = []
    browser = FunctionTool(func=lambda task: opened.append(task) or {"status": "ok"})
    browser.name = "complete_browser_task"
    search = FunctionTool(func=lambda query: searches.append(query) or {"results": []})
    search.name = "searxng_search"
    root.sub_agents[1].tools = [browser, search]

    result = await run_pipeline(root, "Find evidence for this source", session_service=InMemorySessionService())

    assert opened == ["Open https://doi.org/10.1234/abc"]
    assert searches == ["find another copy of Known source DOI 10.1234/abc"]
    ledger = result.state["__fedotmas_research_candidates"]["extractor"]
    assert any(item["doi"] == "10.1234/abc" and item["url"] == "https://doi.org/10.1234/abc" for item in ledger)
