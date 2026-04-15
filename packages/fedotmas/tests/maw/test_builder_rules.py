"""Builder edge-case tests — descriptor-based builder logic."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from pydantic import ValidationError

from fedotmas._settings import ModelConfig
from fedotmas.interfaces.agent import (
    AgentDescriptor,
    LoopDescriptor,
    ParallelDescriptor,
    SequentialDescriptor,
)
from fedotmas.maw.builder import (
    _find_exit_loop_agent,
    _resolve_model,
    build,
)
from fedotmas.maw.models import MAWAgentConfig, MAWConfig


# ---- Rules 1-3: text normalization (via MAWAgentConfig model_validator) ----


class TestNormalizeAngleBrackets:
    """Rule 1: <var> -> {var} (via MAWAgentConfig)."""

    def _make(self, instruction: str) -> MAWAgentConfig:
        return MAWAgentConfig(name="t", instruction=instruction, output_key="k")

    def test_single_var(self):
        assert self._make("<var>").instruction == "{var?}"

    def test_multiple_vars(self):
        assert self._make("<a> and <b>").instruction == "{a?} and {b?}"

    def test_no_vars(self):
        assert self._make("plain text").instruction == "plain text"


class TestMakeVarsOptional:
    """Rule 2: {var} -> {var?} (via MAWAgentConfig)."""

    def _make(self, instruction: str) -> MAWAgentConfig:
        return MAWAgentConfig(name="t", instruction=instruction, output_key="k")

    def test_single_var(self):
        assert self._make("{foo}").instruction == "{foo?}"

    def test_multiple_vars(self):
        assert self._make("{a} then {b}").instruction == "{a?} then {b?}"

    def test_no_vars(self):
        assert self._make("plain text").instruction == "plain text"

    def test_already_optional_stays(self):
        result = self._make("{var?}").instruction
        assert result == "{var?}"


class TestAngleThenOptionalCombo:
    """Rule 3: <var> -> {var} -> {var?} chain (via MAWAgentConfig)."""

    def test_chain(self):
        cfg = MAWAgentConfig(
            name="t",
            instruction="Process <input> and <context>",
            output_key="k",
        )
        assert cfg.instruction == "Process {input?} and {context?}"


# ---- Rules 4-5: model normalization (via MAWAgentConfig) ----


class TestNormalizeModelNameDefault:
    """Rule 4: None/empty model -> stays None (resolved later by _resolve_model)."""

    def test_none_model(self):
        cfg = MAWAgentConfig(name="t", instruction="x", output_key="k", model=None)
        assert cfg.model is None

    def test_resolve_none_gives_default(self):
        result = _resolve_model(None, None)
        assert result == "openai/gpt-oss-120b"


class TestNormalizeModelNamePrefix:
    """Rule 5: bare model name without provider prefix -> ValueError."""

    def test_bare_name(self):
        with pytest.raises(ValidationError, match="must include a provider prefix"):
            MAWAgentConfig(name="t", instruction="x", output_key="k", model="gpt-4o")

    def test_already_prefixed(self):
        cfg = MAWAgentConfig(
            name="t", instruction="x", output_key="k", model="openai/gpt-4o"
        )
        assert cfg.model == "openai/gpt-4o"

    def test_other_provider(self):
        cfg = MAWAgentConfig(
            name="t", instruction="x", output_key="k", model="gemini/flash"
        )
        assert cfg.model == "gemini/flash"


# ---- Rules 6-7: _resolve_model ----


class TestResolveModelCustomEndpoint:
    """Rule 6: model in worker_models -> returns the ModelConfig."""

    def test_custom_endpoint(self):
        cfg = ModelConfig(model="my-model", api_base="http://localhost:9090")
        result = _resolve_model("my-model", {"my-model": cfg})
        assert result is cfg

    def test_custom_with_api_key(self):
        cfg = ModelConfig(model="m", api_base="http://x", api_key="sk-123")
        result = _resolve_model("m", {"m": cfg})
        assert result is cfg


class TestResolveModelNoCustom:
    """Rule 7: model not in worker_models -> plain string."""

    def test_not_in_registry(self):
        other = ModelConfig(model="other")
        result = _resolve_model("openai/gpt-4o", {"other": other})
        assert result == "openai/gpt-4o"


# ---- Rules 8-10: _find_exit_loop_agent ----


class TestFindExitLoopAgent:
    """Rules 8-10: exit_loop agent discovery in loop children."""

    def _make_agent(self, name: str) -> AgentDescriptor:
        return AgentDescriptor(name=name, instruction="do stuff", output_key="k")

    def _make_seq(self, name: str) -> SequentialDescriptor:
        return SequentialDescriptor(name=name, children=[])

    def test_finds_last_agent(self):
        """Rule 8: last AgentDescriptor is selected."""
        a = self._make_agent("a")
        result = _find_exit_loop_agent([a])
        assert result == "a"

    def test_skips_non_agent(self):
        """Rule 10: SequentialDescriptor is not selected."""
        seq = self._make_seq("s")
        result = _find_exit_loop_agent([seq])
        assert result is None

    def test_finds_last_llm(self):
        """exit_loop goes to the *last* AgentDescriptor only."""
        a1 = self._make_agent("first")
        a2 = self._make_agent("second")
        result = _find_exit_loop_agent([a1, a2])
        assert result == "second"

    def test_empty_list_returns_none(self):
        """No children -> None."""
        result = _find_exit_loop_agent([])
        assert result is None


# ---- Rules 11-15: build() ----


class TestBuildSequentialTree:
    """Rule 11: MAWConfig -> SequentialDescriptor with children."""

    @patch("fedotmas.maw.builder.create_toolset", return_value=[])
    def test_sequential(self, _mock_toolset, simple_pipeline_config):
        root = build(simple_pipeline_config)

        assert isinstance(root, SequentialDescriptor)
        assert len(root.children) == 2
        assert root.children[0].name == "alpha"
        assert root.children[1].name == "beta"


class TestBuildParallelTree:
    """Rule 12: parallel type -> ParallelDescriptor."""

    @patch("fedotmas.maw.builder.create_toolset", return_value=[])
    def test_parallel(self, _mock_toolset):
        config = MAWConfig.model_validate(
            {
                "agents": [
                    {"name": "a", "instruction": "do a", "output_key": "oa"},
                    {"name": "b", "instruction": "do b", "output_key": "ob"},
                ],
                "pipeline": {
                    "type": "parallel",
                    "children": [
                        {"type": "agent", "agent_name": "a"},
                        {"type": "agent", "agent_name": "b"},
                    ],
                },
            }
        )
        root = build(config)

        assert isinstance(root, ParallelDescriptor)
        assert len(root.children) == 2


class TestBuildNestedSeqPar:
    """Rule 13: nested sequential -> parallel."""

    @patch("fedotmas.maw.builder.create_toolset", return_value=[])
    def test_nested(self, _mock_toolset):
        config = MAWConfig.model_validate(
            {
                "agents": [
                    {"name": "a", "instruction": "do a", "output_key": "oa"},
                    {"name": "b", "instruction": "do b", "output_key": "ob"},
                    {"name": "c", "instruction": "do c", "output_key": "oc"},
                ],
                "pipeline": {
                    "type": "sequential",
                    "children": [
                        {"type": "agent", "agent_name": "a"},
                        {
                            "type": "parallel",
                            "children": [
                                {"type": "agent", "agent_name": "b"},
                                {"type": "agent", "agent_name": "c"},
                            ],
                        },
                    ],
                },
            }
        )
        root = build(config)

        assert isinstance(root, SequentialDescriptor)
        assert isinstance(root.children[1], ParallelDescriptor)
        assert len(root.children[1].children) == 2


class TestBuildLoopMaxIterations:
    """Rule 14: loop with explicit max_iterations."""

    @patch("fedotmas.maw.builder.create_toolset", return_value=[])
    def test_loop_explicit(self, _mock_toolset):
        config = MAWConfig.model_validate(
            {
                "agents": [
                    {
                        "name": "worker",
                        "instruction": "iterate",
                        "output_key": "result",
                    },
                ],
                "pipeline": {
                    "type": "loop",
                    "max_iterations": 5,
                    "children": [
                        {"type": "agent", "agent_name": "worker"},
                    ],
                },
            }
        )
        root = build(config)

        assert isinstance(root, LoopDescriptor)
        assert root.max_iterations == 5


class TestBuildLoopDefaultMaxIterations:
    """Rule 15: loop without max_iterations -> settings default."""

    @patch("fedotmas.maw.builder.create_toolset", return_value=[])
    @patch("fedotmas.maw.builder.get_max_loop_iterations", return_value=10)
    def test_loop_default(self, _mock_max, _mock_toolset):
        config = MAWConfig.model_validate(
            {
                "agents": [
                    {
                        "name": "worker",
                        "instruction": "iterate",
                        "output_key": "result",
                    },
                ],
                "pipeline": {
                    "type": "loop",
                    "children": [
                        {"type": "agent", "agent_name": "worker"},
                    ],
                },
            }
        )
        root = build(config)

        assert isinstance(root, LoopDescriptor)
        assert root.max_iterations == 10
