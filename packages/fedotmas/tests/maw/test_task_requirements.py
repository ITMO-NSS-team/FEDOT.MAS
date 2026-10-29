from unittest.mock import MagicMock

import pytest
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from pydantic import Field

from fedotmas.core.runner import run_pipeline
from fedotmas.maw import builder
from fedotmas.maw.models import MAWConfig


class Capture(BaseLlm):
    model: str = "offline"
    requests: list = Field(default_factory=list)

    async def generate_content_async(self, request, stream=False):
        self.requests.append(request)
        yield LlmResponse(
            content=types.Content(role="model", parts=[types.Part(text="done")])
        )


def instruction_text(request):
    instruction = request.config.system_instruction
    return (
        instruction
        if isinstance(instruction, str)
        else "\n".join(p.text or "" for p in instruction.parts)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query",
    [
        "How many thousand hours would the trip take? Round to the nearest thousand hours.",
        "Who was the Prime Minister of the first place in Esther in April 1977?",
        'Read this literally: {unknown_key} and {"unit":"thousand hours"}.',
    ],
)
async def test_original_task_reaches_every_model_request_without_generated_reference(
    monkeypatch, query
):
    cfg = MAWConfig.model_validate(
        {
            "agents": [
                {
                    "name": "calculator",
                    "instruction": "Compute rounded_hours in base units.",
                    "output_key": "calculation",
                },
                {
                    "name": "answerer",
                    "instruction": "Return rounded_hours, country and date from {calculation}.",
                    "output_key": "answer",
                },
            ],
            "pipeline": {
                "type": "sequential",
                "children": [{"agent_name": "calculator"}, {"agent_name": "answerer"}],
            },
        }
    )
    llms = [Capture(), Capture()]
    to_build = iter(llms)
    monkeypatch.setattr(builder, "_resolve_llm", lambda *_: next(to_build))
    initial_state = {"user_query": "A stale checkpoint request", "preserved": 7}
    tree = builder.build(
        cfg,
        autonomous=False,
        final_answer_contract="Return only <solution>the requested answer</solution>",
    )
    result = await run_pipeline(tree, query, initial_state=initial_state)
    for llm in llms:
        text = instruction_text(llm.requests[0])
        assert f"ORIGINAL USER TASK (verbatim):\n{query}" in text
        assert "TASK FIDELITY" in text
        assert "unsupported estimate is missing evidence" in text
        assert "A stale checkpoint request" not in text
    assert "terminal answer stage" not in instruction_text(llms[0].requests[0])
    assert "extra deliverables in generated instructions" in instruction_text(
        llms[1].requests[0]
    )
    assert result.state["user_query"] == query
    assert result.state["preserved"] == 7
    assert initial_state["user_query"] == "A stale checkpoint request"


def test_generated_output_cannot_overwrite_original_task():
    with pytest.raises(ValueError, match="reserved"):
        MAWConfig.model_validate(
            {
                "agents": [
                    {
                        "name": "writer",
                        "instruction": "Write",
                        "output_key": "user_query",
                    }
                ],
                "pipeline": {"agent_name": "writer"},
            }
        )


@pytest.mark.asyncio
async def test_repair_can_enable_inspection_and_runtime_respects_the_new_mode(
    monkeypatch,
):
    from fedotmas.control.fixes._fix_agent_configs import (
        AVAILABLE_TOOLS_STATE_KEY,
        repair_agent_configs,
    )
    from google.adk.tools import FunctionTool

    cfg = MAWConfig.model_validate(
        {
            "agents": [
                {
                    "name": "researcher",
                    "instruction": "Discover",
                    "output_key": "answer",
                    "research_mode": "discovery_only",
                }
            ],
            "pipeline": {"agent_name": "researcher"},
        }
    )
    ctx = MagicMock()
    ctx.state = {
        "config": cfg.model_dump_json(),
        AVAILABLE_TOOLS_STATE_KEY: ["web-scraping"],
    }
    result = await repair_agent_configs(
        ctx,
        [
            {
                "agent_name": "researcher",
                "tools": ["web-scraping"],
                "research_mode": "mixed",
                "instruction": "Discover and inspect the requested source",
            }
        ],
        "Inspection is needed",
    )
    assert result.startswith("Updated"), result
    repaired = MAWConfig.model_validate_json(ctx.state["config"])
    monkeypatch.setattr(builder, "_resolve_llm", lambda *_: Capture())
    monkeypatch.setattr(
        builder, "create_toolset", lambda *_, **__: FunctionTool(lambda url: "page")
    )
    html = FunctionTool(lambda url: "page")
    html.name = "web_scraping_html"
    for agent_cfg, allowed in [(cfg.agents[0], False), (repaired.agents[0], True)]:
        agent = builder._build_llm_agent(agent_cfg, None, None, autonomous=False)
        call_context = MagicMock()
        call_context.state = {}
        await agent.before_agent_callback(call_context)
        blocked = await agent.before_tool_callback(
            html, {"url": "https://example.org"}, call_context
        )
        assert (blocked is None) == allowed
