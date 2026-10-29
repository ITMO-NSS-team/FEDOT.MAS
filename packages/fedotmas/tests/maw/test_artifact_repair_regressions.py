import json
from unittest.mock import MagicMock

import pytest
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from pydantic import Field

from fedotmas.maw import builder
from fedotmas.maw._artifact_repair import apply_field_copies
from fedotmas.maw.models import ArtifactContract, MAWAgentConfig


class RepairLlm(BaseLlm):
    model: str = "offline-test"
    answer: str = ""
    requests: list = Field(default_factory=list)

    async def generate_content_async(self, request, stream=False):
        self.requests.append(request)
        yield LlmResponse(
            content=types.Content(
                role="model", parts=[types.Part.from_text(text=self.answer)]
            )
        )


async def repair(monkeypatch, original, fields, answer):
    llm = RepairLlm(answer=answer)
    monkeypatch.setattr(builder, "_resolve_llm", lambda *_: llm)
    cfg = MAWAgentConfig(
        name="worker",
        instruction="Return the artifact",
        output_key="out",
        output_contract=ArtifactContract(required_fields=fields),
    )
    agent = builder._build_llm_agent(cfg, None, None, autonomous=False)
    state = {"out": original}
    ctx = MagicMock()
    ctx.state = state
    ctx._invocation_context.session.state = state
    await agent.after_agent_callback(ctx)
    return state, llm


@pytest.mark.asyncio
async def test_missing_outer_brace_keeps_all_candidates_without_llm(monkeypatch):
    data = {
        "candidates": [{"url": f"https://example.org/{i}"} for i in range(3)],
        "notes": "Do not drop any candidate",
    }
    state, llm = await repair(monkeypatch, json.dumps(data)[:-1], ["candidates"], "")
    assert json.loads(state["out"]) == data
    assert not llm.requests
    assert not state["_fedotmas_execution"].get("handoff_issues")


@pytest.mark.asyncio
async def test_copy_plan_preserves_full_strings_lists_and_original_fields(monkeypatch):
    data = {
        "result": list(range(30)),
        "notes": "full original evidence " * 500,
        "optional": {"quote": "keep this", "more": list(range(20))},
    }
    state, llm = await repair(
        monkeypatch,
        json.dumps(data),
        ["answer"],
        '{"field_copies":[{"from":"/result","to":"/answer"}]}',
    )
    assert json.loads(state["out"]) == {**data, "answer": data["result"]}
    assert len(llm.requests) == 1
    assert not llm.requests[0].config.tools
    assert len(llm.requests[0].contents[0].parts[0].text) < 12000
    assert not state["_fedotmas_execution"].get("handoff_issues")


@pytest.mark.asyncio
async def test_copy_plan_can_wrap_original_without_reconstructing_values(monkeypatch):
    data = {"regions": list(range(20)), "notes": "source " * 300}
    state, _ = await repair(
        monkeypatch,
        json.dumps(data),
        ["data.regions"],
        '{"field_copies":[{"from":"","to":"/data"}]}',
    )
    assert json.loads(state["out"]) == {**data, "data": data}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "plan",
    [
        '{"field_copies":[{"from":"/missing","to":"/answer"}]}',
        '{"field_copies":[{"from":"/result","to":"/answer","value":42}]}',
        '{"field_copies":[{"from":"/result","to":"/answer"}],"answer":42}',
        '{"field_copies":[{"from":"/result","to":"/notes"}]}',
        '{"field_copies":[{"from":"/result","to":"/answer/x"}]}',
        '{"field_copies":[{"from":"/result","to":"/answer"},{"from":"/answer","to":"/evidence"}]}',
    ],
)
async def test_invalid_plan_is_atomic_and_cannot_manufacture_facts(monkeypatch, plan):
    original = '{"result":42,"notes":"must remain"}'
    state, _ = await repair(monkeypatch, original, ["answer", "evidence"], plan)
    assert state["out"] == original
    assert state["_fedotmas_execution"]["handoff_issues"]


@pytest.mark.asyncio
async def test_missing_evidence_is_not_a_rename_of_an_answer(monkeypatch):
    original = '{"answer":42}'
    state, _ = await repair(
        monkeypatch,
        original,
        ["answer", "evidence"],
        '{"field_copies":[{"from":"/answer","to":"/evidence"}]}',
    )
    assert state["out"] == original
    assert state["_fedotmas_execution"]["handoff_issues"]


@pytest.mark.asyncio
async def test_nested_contract_cannot_disguise_answer_as_evidence(monkeypatch):
    original = '{"answer":42}'
    state, _ = await repair(
        monkeypatch,
        original,
        ["data.evidence"],
        '{"field_copies":[{"from":"/answer","to":"/data/evidence"}]}',
    )
    assert state["out"] == original
    assert state["_fedotmas_execution"]["handoff_issues"]


@pytest.mark.asyncio
async def test_existing_array_wrapped_artifact_is_still_accepted(monkeypatch):
    state, llm = await repair(monkeypatch, '[{"answer":42}]', ["answer"], "")
    assert json.loads(state["out"]) == [{"answer": 42}]
    assert not llm.requests
    assert not state.get("_fedotmas_execution", {}).get("handoff_issues")


def test_copy_pointers_preserve_escaped_keys_and_source_array_values():
    source = {"quoted/key": {"~names": ["a", "b"]}}
    result = apply_field_copies(
        source,
        {"field_copies": [{"from": "/quoted~1key/~0names/1", "to": "/answer"}]},
        ArtifactContract(required_fields=["answer"]),
    )
    assert result == {**source, "answer": "b"}
    assert source == {"quoted/key": {"~names": ["a", "b"]}}


@pytest.mark.parametrize(
    "operations",
    [
        [{"from": "/result", "to": "/answer"}],
        [{"from": "/result", "to": "/data/answer"}],
        [{"from": "/result", "to": "/new"}, {"from": "/result", "to": "/new"}],
        [{"from": "/~invalid", "to": "/new"}],
        [{"from": "/result", "to": "/new"}] * 25,
    ],
)
def test_copy_plan_never_mutates_original_or_overwrites_data(operations):
    source = {"result": 42, "answer": 17, "data": "keep"}
    with pytest.raises((ValueError, KeyError)):
        apply_field_copies(
            source,
            {"field_copies": operations},
            ArtifactContract(required_fields=["answer", "data.answer", "new"]),
        )
    assert source == {"result": 42, "answer": 17, "data": "keep"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "original",
    [
        '{"answer":"unterminated',
        '{"answer":[1,2',
        '{"answer":42,',
        '{"answer":',
        '{"answer":{"value":42',
        '{"answer":1} {"answer":2}',
    ],
)
async def test_ambiguous_or_truncated_values_are_not_replaced_with_nested_objects(
    monkeypatch, original
):
    state, llm = await repair(monkeypatch, original, ["answer"], '{"answer":42}')
    assert state["out"] == original
    assert not llm.requests
    assert state["_fedotmas_execution"]["handoff_issues"]
