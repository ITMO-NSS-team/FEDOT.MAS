import json
from unittest.mock import MagicMock

import pytest
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from pydantic import Field

from fedotmas.core.runner import run_pipeline
from fedotmas.maw import builder
from fedotmas.maw.models import MAWConfig
from fedotmas.maw._calculation import evaluate, calculate, CALCULATIONS_KEY


def test_regression_arithmetic_and_decimal_rounding():
    assert evaluate("(4097-C)/(L-2*A)", {"L": 11, "A": 2, "C": 150}, 1) == "563.9"
    assert evaluate("0.1+0.2", {}) == "0.3"
    assert evaluate("x/1000", {"x": "12500"}, 1) == "12.5"


@pytest.mark.parametrize(
    "expression,inputs",
    [
        ("__import__('os').getcwd()", {}),
        ("x.real", {"x": 1}),
        ("[x for x in range(100)]", {}),
        ("2**1000000", {}),
        ("unknown+1", {}),
        ("1/0", {}),
        ("x+1", {"x": "NaN"}),
        ("x+1", {"x": True}),
    ],
)
def test_invalid_calculation_never_creates_success_receipt(expression, inputs):
    ctx = MagicMock(state={}, agent_name="calculator")
    result = calculate(expression, json.dumps(inputs), ctx)
    assert result["status"] == "error"
    assert CALCULATIONS_KEY not in ctx.state


class Scripted(BaseLlm):
    model: str = "offline"
    outputs: list = Field(default_factory=list)
    requests: list = Field(default_factory=list)

    async def generate_content_async(self, request, stream=False):
        output = self.outputs[len(self.requests)]
        self.requests.append(request.model_copy(deep=True))
        if isinstance(output, LlmResponse):
            yield output
            return
        part = (
            types.Part(text=output)
            if isinstance(output, str)
            else types.Part.from_function_call(**output)
        )
        yield LlmResponse(
            content=types.Content(role="model", parts=[part]),
            usage_metadata=types.GenerateContentResponseUsageMetadata(
                prompt_token_count=3, candidates_token_count=2
            ),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("reported", ["563.9", "692.4"])
async def test_numeric_handoff_must_match_executed_result(monkeypatch, reported):
    model = Scripted(
        outputs=[
            {
                "name": "calculate",
                "args": {
                    "expression": "(4097-C)/(L-2*A)",
                    "inputs_json": '{"L":11,"A":2,"C":150}',
                    "decimal_places": 1,
                },
            },
            json.dumps({"value": reported}),
        ]
    )
    monkeypatch.setattr(builder, "_resolve_llm", lambda *_: model)
    cfg = MAWConfig.model_validate(
        {
            "agents": [
                {
                    "name": "calculator",
                    "instruction": "Calculate",
                    "output_key": "out",
                    "calculation_result_field": "value",
                    "output_contract": {"required_fields": ["value"]},
                }
            ],
            "pipeline": {"agent_name": "calculator"},
        }
    )
    result = await run_pipeline(builder.build(cfg), "Evaluate the expression")
    assert result.status == ("completed" if reported == "563.9" else "incomplete")
    assert result.state[CALCULATIONS_KEY]["calculator"][-1]["result"] == "563.9"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "draft,answer",
    [("INT. THE CASTLE - DAY", "THE CASTLE"), ("No original research", "research")],
)
async def test_terminal_research_gets_separate_tools_free_presentation(
    monkeypatch, draft, answer
):
    model = Scripted(outputs=[draft, answer])
    monkeypatch.setattr(builder, "_resolve_llm", lambda *_: model)
    monkeypatch.setattr(builder, "create_toolset", lambda *a, **kw: lambda: "stub")
    cfg = MAWConfig.model_validate(
        {
            "agents": [
                {
                    "name": "researcher",
                    "instruction": "Read sources. Include the whole source line in your answer.",
                    "output_key": "answer",
                    "tools": ["document"],
                }
            ],
            "pipeline": {"agent_name": "researcher"},
        }
    )
    result = await run_pipeline(
        builder.build(cfg, final_answer_contract="Return only the requested words."),
        "Return only the requested words.",
    )
    assert result.state["answer"] == answer
    assert len(model.requests) == 2
    assert "Include the whole source line" in str(
        model.requests[0].config.system_instruction
    )
    assert "Include the whole source line" not in str(model.requests[1])
    assert not model.requests[1].config.tools
    assert result.total_prompt_tokens == 6 and result.total_completion_tokens == 4
    assert result.state["_fedotmas_execution"]["agent_llm_turns"]["researcher"] == 2


@pytest.mark.asyncio
async def test_formatter_does_not_exceed_worker_turn_budget(monkeypatch):
    model = Scripted(outputs=["Supported draft"])
    monkeypatch.setattr(builder, "_resolve_llm", lambda *_: model)
    monkeypatch.setattr(builder, "create_toolset", lambda *a, **kw: lambda: "stub")
    cfg = MAWConfig.model_validate(
        {
            "agents": [
                {
                    "name": "researcher",
                    "instruction": "Read",
                    "output_key": "answer",
                    "tools": ["document"],
                    "max_llm_turns": 1,
                }
            ],
            "pipeline": {"agent_name": "researcher"},
        }
    )
    result = await run_pipeline(
        builder.build(cfg, final_answer_contract="Answer only"), "Question"
    )
    assert result.status == "limited"
    assert len(model.requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "presentation",
    [
        "",
        {"name": "unexpected_tool", "args": {}},
        LlmResponse(
            content=types.Content(role="model", parts=[types.Part(text="Partial")]),
            finish_reason=types.FinishReason.MAX_TOKENS,
        ),
    ],
)
async def test_invalid_presentation_is_not_a_completed_answer(
    monkeypatch, presentation
):
    model = Scripted(outputs=["Supported draft", presentation])
    monkeypatch.setattr(builder, "_resolve_llm", lambda *_: model)
    monkeypatch.setattr(builder, "create_toolset", lambda *a, **kw: lambda: "stub")
    cfg = MAWConfig.model_validate(
        {
            "agents": [
                {
                    "name": "researcher",
                    "instruction": "Read",
                    "output_key": "answer",
                    "tools": ["document"],
                }
            ],
            "pipeline": {"agent_name": "researcher"},
        }
    )
    result = await run_pipeline(
        builder.build(cfg, final_answer_contract="Answer only"), "Question"
    )
    assert result.status == "incomplete"
    assert result.state["answer"].startswith("<abstain>")
    assert (
        result.state["__fedotmas_terminal_research"]["researcher"] == "Supported draft"
    )
