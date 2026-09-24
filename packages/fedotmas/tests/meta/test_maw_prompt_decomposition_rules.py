from __future__ import annotations

from fedotmas.meta.maw_prompts import (
    META_AGENT_SYSTEM_PROMPT,
    PIPELINE_AGENT_SYSTEM_PROMPT,
    POOL_AGENT_SYSTEM_PROMPT,
)


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
