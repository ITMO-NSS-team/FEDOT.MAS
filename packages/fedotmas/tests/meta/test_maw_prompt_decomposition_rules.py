from __future__ import annotations

import re
from unittest.mock import MagicMock

import pytest
from fedotmas.meta._helpers import format_server_descriptions
from fedotmas.meta.maw_prompts import (
    META_AGENT_SYSTEM_PROMPT,
    PIPELINE_AGENT_SYSTEM_PROMPT,
    POOL_AGENT_SYSTEM_PROMPT,
)
from google.adk.utils.instructions_utils import inject_session_state


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "prompt_template",
    [
        META_AGENT_SYSTEM_PROMPT,
        POOL_AGENT_SYSTEM_PROMPT,
        PIPELINE_AGENT_SYSTEM_PROMPT,
    ],
    ids=["single-stage", "pool-stage", "pipeline-stage"],
)
async def test_rendered_meta_prompts_are_safe_with_empty_adk_state(prompt_template):
    """Render the prompts through ADK's own state interpolation path."""
    rendered = prompt_template.substitute(
        mcp_servers_desc=format_server_descriptions(
            {"websearch-searxng": "Search the web"}
        ),
        available_models="- `openai/gpt-4o`",
    )
    live_state_refs = re.findall(r"(?<!\{)\{(\w+)\??\}(?!\})", rendered)

    assert not live_state_refs

    context = MagicMock()
    context.state = {}
    context._invocation_context.session.state = {}
    context._invocation_context.artifact_service = None
    assert await inject_session_state(rendered, context) == rendered


def test_single_stage_prompt_uses_task_driven_team_size_and_specialization():
    prompt = META_AGENT_SYSTEM_PROMPT.template

    assert "Use the minimum sufficient semantic decomposition" in prompt
    assert "Do not create separate agents merely to query two search providers" in prompt
    assert "Keep a task in one agent when it is atomic" in prompt
    assert "tool specialization, independent evidence gathering, context isolation" in prompt
    assert "reduces ambiguity or context mixing" in prompt
    assert "concise, source-backed findings" in prompt
    assert "source_finder -> structured_extractor" in prompt
    assert "Do not force this decomposition for simple lookups" in prompt
    assert "without repeating broad discovery" in prompt
    assert "Use one exact authoritative source when the task permits it" in prompt
    assert "Start simple. Use 1–2 agents" not in prompt


def test_single_stage_prompt_examples_show_dependent_and_independent_work():
    prompt = META_AGENT_SYSTEM_PROMPT.template

    assert "identifier_finder" in prompt
    assert "dependent_researcher" in prompt
    assert '"agent_name": "verifier"' in prompt
    assert "source_A_researcher" in prompt
    assert "source_B_researcher" in prompt
    assert "synthesizer_verifier" in prompt
    assert "Use parallel only for genuinely independent branches" in prompt
    assert "Use this pattern only when the task itself needs independent corroboration" in prompt


def test_two_stage_prompts_share_task_driven_decomposition_rules():
    pool_prompt = POOL_AGENT_SYSTEM_PROMPT.template
    pipeline_prompt = PIPELINE_AGENT_SYSTEM_PROMPT.template

    assert "Use the minimum sufficient semantic decomposition" in pool_prompt
    assert "Do not create separate agents merely to query different providers" in pool_prompt
    assert "independent evidence gathering, context isolation" in pool_prompt
    assert "source_finder" in pool_prompt
    assert "domain_interpreter" in pool_prompt
    assert "substantial extraction benefits from a separate role" in pool_prompt
    assert "Use the minimum sufficient semantic decomposition" in pipeline_prompt
    assert "one researcher can switch providers" in pipeline_prompt
    assert "Use parallel only for genuinely independent branches" in pipeline_prompt
    assert "Do not impose a universal multiple-source evidence rule" in pipeline_prompt
    assert "when the source branches do not need each other's results" in pipeline_prompt
    assert "Do not make provider-specific search workers" in pipeline_prompt


def test_maw_prompts_route_material_computation_to_code_agent():
    for prompt in (
        META_AGENT_SYSTEM_PROMPT.template,
        POOL_AGENT_SYSTEM_PROMPT.template,
        PIPELINE_AGENT_SYSTEM_PROMPT.template,
    ):
        assert "code-agent" in prompt
        assert "document" in prompt
    assert "do not add `code-agent` to every research role by default" in (
        META_AGENT_SYSTEM_PROMPT.template.casefold()
    )


def test_maw_prompts_use_current_browser_and_search_routing():
    for prompt in (
        META_AGENT_SYSTEM_PROMPT.template,
        POOL_AGENT_SYSTEM_PROMPT.template,
        PIPELINE_AGENT_SYSTEM_PROMPT.template,
    ):
        assert "browser-agent" in prompt
        assert "browser-usage" not in prompt
        assert "assign `websearch-tavily`" in prompt
        assert "worker-facing `search` tool" in prompt
        assert "routes Tavily first, then SearXNG internally" in prompt
        assert "web-scraping" in prompt
        assert "do not assign it to every research role" in prompt
        assert "download/document tools" in prompt


def test_contract_prompts_keep_required_fields_minimal():
    for prompt in (
        META_AGENT_SYSTEM_PROMPT.template,
        PIPELINE_AGENT_SYSTEM_PROMPT.template,
    ):
        lowered = prompt.casefold()
        assert "keep `output_contract.required_fields` minimal" in lowered
        assert "research_controller_state` as an additional output field" in lowered


@pytest.mark.parametrize(
    "prompt_template",
    [META_AGENT_SYSTEM_PROMPT, PIPELINE_AGENT_SYSTEM_PROMPT],
    ids=["single-stage", "pipeline-stage"],
)
def test_research_prompts_require_snapshot_controller_loop(prompt_template):
    prompt = prompt_template.template.casefold()

    assert "research-controller" in prompt
    assert "before expensive" in prompt
    assert "after several searches" in prompt
    assert "before final synthesis or handoff" in prompt
    assert "strategy_blocked" in prompt
    assert "research_state" in prompt
