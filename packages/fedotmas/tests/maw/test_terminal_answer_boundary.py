"""Regression cases from terminal answers that leaked intermediate fields."""

import json

import pytest
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from pydantic import Field

from fedotmas.core.runner import run_pipeline
from fedotmas.maw import builder
from fedotmas.maw.handoffs import (
    ABSTENTION_STATE_KEY,
    explicit_abstention_reason,
    is_explicit_abstention,
)
from fedotmas.maw.models import MAWConfig


class Capture(BaseLlm):
    model: str = "offline"
    output: str = "<solution>result</solution>"
    requests: list = Field(default_factory=list)

    async def generate_content_async(self, request, stream=False):
        self.requests.append(request)
        yield LlmResponse(
            content=types.Content(role="model", parts=[types.Part(text=self.output)])
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query,plan,evidence",
    [
        (
            "How many more blocks does BERT base have than the Transformer encoder?",
            "Return BERT blocks, Transformer blocks and their difference.",
            {"bert": 12, "transformer": 6, "difference": 6},
        ),
        (
            "Find the vendor with the lowest revenue/rent ratio. Tell me its type.",
            "Name the vendor and state the type column.",
            {"name": "Carnivore Loan Specialists", "type": "Finance"},
        ),
        (
            "Who was prime minister of the first place in Esther in April 1977?",
            "Name the first place and its prime minister, then include evidence.",
            {"country": "India", "pm": "Morarji Desai"},
        ),
        (
            "Return the country and prime minister, in that order, separated by commas.",
            "Return just the prime minister.",
            {"country": "India", "pm": "Morarji Desai"},
        ),
    ],
)
@pytest.mark.parametrize("contracted_input", [False, True])
async def test_answer_prompt_uses_original_task_and_data_not_generated_deliverables(
    monkeypatch, query, plan, evidence, contracted_input
):
    config = MAWConfig.model_validate(
        {
            "agents": [
                {"name": "source", "instruction": "Find facts.", "output_key": "facts"},
                {
                    "name": "answerer",
                    # A nested reference must preserve the containing artifact too.
                    "instruction": plan + " Read {facts.country}."
                    if not contracted_input
                    else plan,
                    "output_key": "answer",
                    "input_requirements": [
                        {
                            "source_key": "facts",
                            "required_fields": [next(iter(evidence))],
                            "identity_fields": [next(iter(evidence))],
                            "purpose": plan,
                        }
                    ]
                    if contracted_input
                    else [],
                },
            ],
            "pipeline": {
                "type": "sequential",
                "children": [{"agent_name": "source"}, {"agent_name": "answerer"}],
            },
        }
    )
    before = config.model_dump()
    source = Capture(output=json.dumps(evidence))
    answerer = Capture()
    models = iter([source, answerer])
    monkeypatch.setattr(builder, "_resolve_llm", lambda *_: next(models))
    result = await run_pipeline(
        builder.build(
            config, autonomous=False, final_answer_contract="Use solution tags."
        ),
        query,
    )
    request = answerer.requests[0]
    text = str(request.config.system_instruction)
    assert plan not in text
    assert query in text
    assert json.dumps(evidence) in text
    assert "Source-to-output field paths" not in text
    assert "Handoff purpose:" not in text
    assert "Preserve explicitly requested lists" in text
    assert result.status == "completed"
    assert len(answerer.requests) == 1  # No additional formatter/model pass.
    assert config.agents[-1].model_dump() == before["agents"][-1]


@pytest.mark.parametrize(
    "value,reason",
    [
        ("<abstain>No source.</abstain>", "No source."),
        (
            "I tried several URLs.\n\n<abstain>No enrollment obtained.</abstain>",
            "No enrollment obtained.",
        ),
        ("<solution><abstain>No source.</abstain></solution>", "No source."),
        ("<solution>42</solution>\n<abstain>Unsupported.</abstain>", "Unsupported."),
        ({"status": "abstained", "reason": "No source."}, "No source."),
        ('{"status":"abstained","reason":"No source."}', "No source."),
        (
            "Earlier: <abstain>No source.</abstain> Recovered: <solution>42</solution>",
            None,
        ),
        ('The documentation uses "<abstain>reason</abstain>" as an example.', None),
        ("Example:\n```\n<abstain>reason</abstain>\n```", None),
        ("Earlier: <abstain>No source.</abstain> Now a supported answer exists.", None),
        ("<solution>42</solution>", None),
        (None, None),
    ],
)
def test_terminal_abstention_boundaries(value, reason):
    assert explicit_abstention_reason(value) == reason
    assert is_explicit_abstention(value) == (reason is not None)


@pytest.mark.asyncio
@pytest.mark.parametrize("recovered", [False, True])
async def test_prose_then_abstention_reaches_pipeline_status_and_can_recover(
    monkeypatch, recovered
):
    config = MAWConfig.model_validate(
        {
            "agents": [
                {
                    "name": "answerer",
                    "instruction": "Find enrollment.",
                    "output_key": "answer",
                }
            ],
            "pipeline": {"agent_name": "answerer"},
        }
    )
    model = Capture(
        output="Inspection failed.\n<abstain>Enrollment unavailable.</abstain>"
    )
    monkeypatch.setattr(builder, "_resolve_llm", lambda *_: model)
    tree = builder.build(
        config, autonomous=False, final_answer_contract="Use solution tags."
    )
    result = await run_pipeline(tree, "Find the enrollment count.")
    assert result.status == "incomplete"
    assert result.state[ABSTENTION_STATE_KEY] == {
        "status": "abstained",
        "agent": "answerer",
        "reason": "Enrollment unavailable.",
    }
    if recovered:
        model.output = "<solution>90</solution>"
        result = await run_pipeline(
            tree, "Find the enrollment count.", initial_state=result.state
        )
        assert result.status == "completed"
        assert result.state[ABSTENTION_STATE_KEY].get("status") not in {
            "abstained",
            "unresolved",
        }


@pytest.mark.asyncio
@pytest.mark.parametrize("recovered", [False, True])
async def test_empty_structured_refusal_is_incomplete_and_can_recover(
    monkeypatch, recovered
):
    config = MAWConfig(
        agents=[{"name": "answerer", "instruction": "Answer.", "output_key": "answer"}],
        pipeline={"agent_name": "answerer"},
    )
    model = Capture(
        output=json.dumps(
            {
                "first_date": None,
                "first_year": None,
                "first_close_price": None,
                "evidence_url": ["https://example.org/history"],
                "evidence_excerpt": "403: Access denied",
                "notes": "Abstained: live and archived sources were inaccessible.",
            }
        )
    )
    monkeypatch.setattr(builder, "_resolve_llm", lambda *_: model)
    tree = builder.build(config, autonomous=False)
    result = await run_pipeline(tree, "Find the first year.")
    assert result.status == "incomplete"
    assert result.state[ABSTENTION_STATE_KEY]["status"] == "unresolved"
    if recovered:
        model.output = "<solution>2018</solution>"
        result = await run_pipeline(
            tree, "Find the first year.", initial_state=result.state
        )
        assert result.status == "completed"


@pytest.mark.asyncio
async def test_single_agent_without_tools_uses_user_question(monkeypatch):
    config = MAWConfig.model_validate(
        {
            "agents": [
                {
                    "name": "answerer",
                    "instruction": "Return three numbers.",
                    "output_key": "answer",
                }
            ],
            "pipeline": {"agent_name": "answerer"},
        }
    )
    llm = Capture()
    monkeypatch.setattr(builder, "_resolve_llm", lambda *_: llm)
    await run_pipeline(
        builder.build(config, final_answer_contract="Use solution tags."),
        'Compute the difference only; preserve literal {"x": 12}.',
    )
    text = str(llm.requests[0].config.system_instruction)
    assert "Return three numbers." not in text
    assert 'preserve literal {"x": 12}' in text
