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

    assert "Do not optimize for the smallest possible team" in prompt
    assert "genuinely atomic task" in prompt
    assert "distinct evidence sources, modalities, tools" in prompt
    assert "reduces ambiguity or context mixing" in prompt
    assert "concise, source-backed findings" in prompt
    assert "source finding, domain interpretation, identifier resolution" in prompt
    assert (
        "source_finder -> domain_interpreter -> identifier_resolver -> verifier"
        in prompt
    )
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


def test_two_stage_prompts_share_task_driven_decomposition_rules():
    pool_prompt = POOL_AGENT_SYSTEM_PROMPT.template
    pipeline_prompt = PIPELINE_AGENT_SYSTEM_PROMPT.template

    assert "Do not optimize for the smallest possible team" in pool_prompt
    assert "semantic ambiguity or context mixing" in pool_prompt
    assert "source_finder" in pool_prompt
    assert "domain_interpreter" in pool_prompt
    assert "identifier_resolver -> verifier" in pool_prompt
    assert "Do not optimize for the smallest possible team" not in pipeline_prompt
    assert "Before using `parallel`" in pipeline_prompt
    assert "identifier_finder" in pipeline_prompt
    assert "dependent_researcher" in pipeline_prompt
    assert "source_A_researcher" in pipeline_prompt
    assert "source_B_researcher" in pipeline_prompt
    assert (
        "[source_A_researcher, source_B_researcher] -> synthesizer/verifier"
        in pipeline_prompt
    )


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
        assert "websearch-searxng" in prompt
        assert "websearch-tavily" in prompt
        assert "web-scraping" in prompt
        assert "empty or poor results" in prompt


@pytest.mark.parametrize(
    "prompt_template",
    [META_AGENT_SYSTEM_PROMPT, POOL_AGENT_SYSTEM_PROMPT, PIPELINE_AGENT_SYSTEM_PROMPT],
    ids=["single-stage", "pool-stage", "pipeline-stage"],
)
def test_research_prompts_require_snapshot_controller_loop(prompt_template):
    prompt = prompt_template.template.casefold()

    assert "research-controller" in prompt
    assert "before expensive" in prompt
    assert "after several searches" in prompt
    assert "before final synthesis or handoff" in prompt
    assert "strategy_blocked" in prompt
    assert "research_state" in prompt
