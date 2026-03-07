"""Pydantic validation rules for pipeline models — no mocks needed."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from fedotmas.maw.models import (
    AgentPoolConfig,
    MAWConfig,
    MAWStepConfig,
)


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
        step = MAWStepConfig(**{"agent_name": "x", "type": "agent"})
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
