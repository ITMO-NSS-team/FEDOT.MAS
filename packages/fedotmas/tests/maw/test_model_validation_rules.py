"""Pydantic validation rules for pipeline models — no mocks needed."""

from __future__ import annotations

import pytest
from fedotmas.maw.maw import (
    _drop_generated_token_budgets,
    _normalize_generated_agent_names,
    _normalize_generated_research_policies,
)
from fedotmas.maw.models import (
    AgentPoolConfig,
    ArtifactRequirement,
    MAWAgentConfig,
    MAWConfig,
    MAWStepConfig,
)
from pydantic import ValidationError


class TestDuplicateAgentNames:
    """Rule 1: MAWConfig rejects duplicate agent names."""

    def test_duplicate_names_rejected(self):
        with pytest.raises(ValidationError, match="Duplicate agent name"):
            MAWConfig(
                agents=[
                    {"name": "a", "instruction": "x", "output_key": "k1"},
                    {"name": "a", "instruction": "y", "output_key": "k2"},
                ],
                pipeline={
                    "type": "sequential",
                    "children": [
                        {"type": "agent", "agent_name": "a"},
                    ],
                },
            )


class TestDuplicateOutputKey:
    """Rule 2: MAWConfig rejects duplicate output_key."""

    def test_duplicate_output_keys_rejected(self):
        with pytest.raises(ValidationError, match="Duplicate output_key"):
            MAWConfig(
                agents=[
                    {"name": "a", "instruction": "x", "output_key": "same"},
                    {"name": "b", "instruction": "y", "output_key": "same"},
                ],
                pipeline={
                    "type": "sequential",
                    "children": [
                        {"type": "agent", "agent_name": "a"},
                        {"type": "agent", "agent_name": "b"},
                    ],
                },
            )


class TestUnknownAgentRef:
    """Rule 3: Pipeline referencing a non-existent agent is rejected."""

    def test_unknown_agent_rejected(self):
        with pytest.raises(ValidationError, match="unknown agent"):
            MAWConfig(
                agents=[
                    {"name": "a", "instruction": "x", "output_key": "k1"},
                ],
                pipeline={"type": "agent", "agent_name": "ghost"},
            )


class TestAutoInferType:
    """Rules 4-5: MAWStepConfig auto-infers type from fields."""

    def test_infer_agent_type(self):
        step = MAWStepConfig(agent_name="x", type="agent")
        assert step.type == "agent"

    def test_infer_agent_type_from_agent_name(self):
        step = MAWStepConfig.model_validate({"agent_name": "x"})
        assert step.type == "agent"

    def test_infer_sequential_from_children(self):
        step = MAWStepConfig.model_validate(
            {
                "children": [{"type": "agent", "agent_name": "x"}],
            }
        )
        assert step.type == "sequential"


class TestPoolDuplicateNames:
    """Rule 6: AgentPoolConfig rejects duplicate names."""

    def test_pool_duplicate_names_rejected(self):
        with pytest.raises(ValidationError, match="Duplicate agent name in pool"):
            AgentPoolConfig(
                agents=[
                    {"name": "a", "instruction": "x"},
                    {"name": "a", "instruction": "y"},
                ]
            )


class TestSingleAgentAutoFill:
    """Rule 7: Single agent → pipeline node gets agent_name automatically."""

    def test_single_agent_auto_fill(self, single_agent_data):
        config = MAWConfig.model_validate(single_agent_data)
        assert config.pipeline.agent_name == "solver"


def test_generated_names_normalize_and_update_pipeline_and_terminal_refs():
    config = MAWConfig.model_validate(
        {
            "agents": [
                {"name": "2 scout", "instruction": "x", "output_key": "a"},
                {"name": "2-scout", "instruction": "y", "output_key": "b"},
                {"name": "final answer!", "instruction": "z", "output_key": "c"},
            ],
            "final_answer_agent": "final answer!",
            "pipeline": {
                "type": "sequential",
                "children": [
                    {"type": "agent", "agent_name": "2 scout"},
                    {"type": "agent", "agent_name": "2-scout"},
                    {"type": "agent", "agent_name": "final answer!"},
                ],
            },
        }
    )

    normalized = _normalize_generated_agent_names(config)

    assert [agent.name for agent in normalized.agents] == [
        "_2_scout",
        "_2_scout_2",
        "final_answer_",
    ]
    assert normalized.final_answer_agent == "final_answer_"
    assert [node.agent_name for node in normalized.pipeline.children] == [
        "_2_scout",
        "_2_scout_2",
        "final_answer_",
    ]


def test_name_normalization_leaves_caller_supplied_names_untouched():
    config = MAWConfig.model_validate(
        {
            "agents": [
                {"name": "2 caller", "instruction": "x", "output_key": "caller"},
                {"name": "2-caller", "instruction": "y", "output_key": "generated"},
            ],
            "pipeline": {
                "type": "sequential",
                "children": [
                    {"type": "agent", "agent_name": "2 caller"},
                    {"type": "agent", "agent_name": "2-caller"},
                ],
            },
        }
    )

    normalized = _normalize_generated_agent_names(config, preserved_names={"2 caller"})

    assert [agent.name for agent in normalized.agents] == ["2 caller", "_2_caller"]


class TestNonLeafWithoutChildren:
    """Rule 8: sequential/parallel/loop without children → ValidationError."""

    @pytest.mark.parametrize("step_type", ["sequential", "parallel", "loop"])
    def test_non_leaf_no_children(self, step_type):
        with pytest.raises(ValidationError, match="must have at least one child"):
            MAWConfig(
                agents=[
                    {"name": "a", "instruction": "x", "output_key": "k1"},
                ],
                pipeline={"type": step_type, "children": []},
            )


class TestModelValidation:
    """Rule 9: Bare model name without provider prefix → ValueError."""

    def test_bare_model_rejected(self):
        with pytest.raises(ValidationError, match="must include a provider prefix"):
            MAWAgentConfig(name="a", instruction="x", output_key="k1", model="gpt-4o")

    def test_prefixed_model_accepted(self):
        cfg = MAWAgentConfig(
            name="a", instruction="x", output_key="k1", model="openai/gpt-4o"
        )
        assert cfg.model == "openai/gpt-4o"

    def test_none_model_accepted(self):
        cfg = MAWAgentConfig(name="a", instruction="x", output_key="k1", model=None)
        assert cfg.model is None


class TestInstructionNormalization:
    """Rule 10: state references are normalized without corrupting XML tags."""

    def test_curly_state_reference_becomes_optional(self):
        cfg = MAWAgentConfig(
            name="a",
            instruction="Use {research_result} to answer.",
            output_key="k1",
        )
        assert cfg.instruction == "Use {research_result?} to answer."

    def test_solution_tags_are_preserved(self):
        cfg = MAWAgentConfig(
            name="a",
            instruction="Return only <solution>answer</solution>.",
            output_key="k1",
        )
        assert cfg.instruction == "Return only <solution>answer</solution>."

    def test_known_angle_state_reference_becomes_optional_in_config(self):
        cfg = MAWConfig(
            agents=[
                {
                    "name": "a",
                    "instruction": "Use <user_query> and <research_result>.",
                    "output_key": "research_result",
                },
            ],
            pipeline={"type": "agent", "agent_name": "a"},
        )
        assert cfg.agents[0].instruction == "Use {user_query?} and {research_result?}."

    def test_solution_tags_are_preserved_in_config(self):
        cfg = MAWConfig(
            agents=[
                {
                    "name": "a",
                    "instruction": "Return only <solution>answer</solution>.",
                    "output_key": "solution",
                },
            ],
            pipeline={"type": "agent", "agent_name": "a"},
        )
        assert cfg.agents[0].instruction == "Return only <solution>answer</solution>."


class TestAgentNameChildrenConflict:
    """Rule 11: ambiguous agent_name + children handling."""

    def test_untyped_agent_name_and_children_rejected(self):
        with pytest.raises(ValidationError, match="Cannot specify both"):
            MAWStepConfig(
                agent_name="a",
                children=[MAWStepConfig(type="agent", agent_name="b")],
            )

    def test_typed_sequential_drops_agent_name(self):
        step = MAWStepConfig.model_validate(
            {
                "type": "sequential",
                "agent_name": "ignored",
                "children": [{"type": "agent", "agent_name": "a"}],
                "max_iterations": 0,
            }
        )
        assert step.type == "sequential"
        assert step.agent_name is None
        assert step.max_iterations is None
        assert step.children[0].agent_name == "a"

    def test_typed_agent_drops_children(self):
        step = MAWStepConfig.model_validate(
            {
                "type": "agent",
                "agent_name": "a",
                "children": [{"type": "agent", "agent_name": "ignored"}],
            }
        )
        assert step.type == "agent"
        assert step.agent_name == "a"
        assert step.children == []


class TestGeneratedTokenBudget:
    """Nothing asks the meta-agent for a cap, so nothing it returns is one."""

    def _config(self, max_output_tokens):
        return MAWConfig(
            agents=[
                MAWAgentConfig(
                    name="calculator",
                    instruction="Compute things.",
                    output_key="calc",
                    max_output_tokens=max_output_tokens,
                )
            ],
            pipeline=MAWStepConfig(type="agent", agent_name="calculator"),
        )

    def test_a_small_budget_is_dropped_not_clamped(self):
        """Clamping made the floor the cap every generated agent hit."""
        config = self._config(2000)
        _drop_generated_token_budgets(config)

        assert config.agents[0].max_output_tokens is None

    def test_a_roomier_budget_is_dropped_too(self):
        """A larger guess is still a guess: no threshold earns trust here."""
        config = self._config(8000)
        _drop_generated_token_budgets(config)

        assert config.agents[0].max_output_tokens is None

    def test_unset_stays_unset(self):
        config = self._config(None)
        _drop_generated_token_budgets(config)

        assert config.agents[0].max_output_tokens is None

    def test_generated_turn_limit_is_discarded(self):
        config = self._config(None)
        config.agents[0].max_llm_turns = 4
        _drop_generated_token_budgets(config)

        assert config.agents[0].max_llm_turns is None

    def test_handwritten_turn_limit_is_preserved_without_generated_cleanup(self):
        config = self._config(None)
        config.agents[0].max_llm_turns = 4

        assert config.agents[0].max_llm_turns == 4

    def test_evidence_first_without_inputs_becomes_independent(self):
        config = self._config(None)
        config.agents[0].research_policy = "evidence_first"
        _drop_generated_token_budgets(config)

        assert config.agents[0].research_policy == "independent"

    def test_evidence_first_first_stage_with_non_upstream_requirement_is_independent(self):
        config = MAWConfig(
            agents=[
                MAWAgentConfig(
                    name="first",
                    instruction="Discover evidence.",
                    output_key="first_output",
                    research_policy="evidence_first",
                    input_requirements=[
                        ArtifactRequirement(
                            source_key="later_output", required_fields=["evidence"]
                        )
                    ],
                ),
                MAWAgentConfig(
                    name="later",
                    instruction="Inspect evidence.",
                    output_key="later_output",
                ),
            ],
            pipeline=MAWStepConfig(
                type="sequential",
                children=[
                    MAWStepConfig(type="agent", agent_name="first"),
                    MAWStepConfig(type="agent", agent_name="later"),
                ],
            ),
        )

        _normalize_generated_research_policies(config)

        assert config.agents[0].research_policy == "independent"
