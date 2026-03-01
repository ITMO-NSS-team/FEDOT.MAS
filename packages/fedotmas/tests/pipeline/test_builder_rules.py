"""Builder edge-case tests — patch ADK constructors, test logic."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from fedotmas.config.settings import ModelConfig
from fedotmas.pipeline.builder import (
    _inject_exit_loop,
    _make_llm,
    _make_vars_optional,
    _normalize_angle_brackets,
    build,
)
from fedotmas.pipeline.models import PipelineConfig


# ---- Rules 1-3: text normalization (pure functions) ----


class TestNormalizeAngleBrackets:
    """Rule 1: <var> → {var}."""

    def test_single_var(self):
        assert _normalize_angle_brackets("<var>") == "{var}"

    def test_multiple_vars(self):
        assert _normalize_angle_brackets("<a> and <b>") == "{a} and {b}"

    def test_no_vars(self):
        assert _normalize_angle_brackets("plain text") == "plain text"


class TestMakeVarsOptional:
    """Rule 2: {var} → {var?}."""

    def test_single_var(self):
        assert _make_vars_optional("{foo}") == "{foo?}"

    def test_multiple_vars(self):
        assert _make_vars_optional("{a} then {b}") == "{a?} then {b?}"

    def test_no_vars(self):
        assert _make_vars_optional("plain text") == "plain text"

    def test_already_optional_gets_doubled(self):
        # Edge: if already optional, regex still matches the non-? part
        # {var?} has no \w match for '?' so it stays as-is
        result = _make_vars_optional("{var?}")
        # The regex \w+ won't match "var?" so it passes through unchanged
        assert result == "{var?}"


class TestAngleThenOptionalCombo:
    """Rule 3: <var> → {var} → {var?} chain."""

    def test_chain(self):
        raw = "Process <input> and <context>"
        step1 = _normalize_angle_brackets(raw)
        assert step1 == "Process {input} and {context}"
        step2 = _make_vars_optional(step1)
        assert step2 == "Process {input?} and {context?}"


# ---- Rules 4-7: _make_llm ----


class TestMakeLlmDefaultFallback:
    """Rule 4: None model → default."""

    def test_none_model(self):
        assert _make_llm(None, None) == "openai/gpt-oss-120b"

    def test_empty_string(self):
        assert _make_llm("", None) == "openai/gpt-oss-120b"


class TestMakeLlmAutoPrefix:
    """Rule 5: bare model name gets openai/ prefix."""

    def test_bare_name(self):
        assert _make_llm("gpt-4o", None) == "openai/gpt-4o"

    def test_already_prefixed(self):
        assert _make_llm("openai/gpt-4o", None) == "openai/gpt-4o"

    def test_other_provider(self):
        assert _make_llm("gemini/flash", None) == "gemini/flash"


class TestMakeLlmCustomEndpoint:
    """Rule 6: model with api_base → LiteLlm instance."""

    @patch("fedotmas.pipeline.builder.LiteLlm")
    def test_custom_endpoint(self, mock_lite):
        cfg = ModelConfig(model="my-model", api_base="http://localhost:8080")
        _make_llm("my-model", {"my-model": cfg})
        mock_lite.assert_called_once_with(model="my-model", api_base="http://localhost:8080")

    @patch("fedotmas.pipeline.builder.LiteLlm")
    def test_custom_with_api_key(self, mock_lite):
        cfg = ModelConfig(model="m", api_base="http://x", api_key="sk-123")
        _make_llm("m", {"m": cfg})
        mock_lite.assert_called_once_with(model="m", api_base="http://x", api_key="sk-123")


class TestMakeLlmNoCustom:
    """Rule 7: model not in worker_models → plain string."""

    def test_not_in_registry(self):
        other = ModelConfig(model="other")
        result = _make_llm("openai/gpt-4o", {"other": other})
        assert result == "openai/gpt-4o"


# ---- Rules 8-10: _inject_exit_loop ----


class TestInjectExitLoop:
    """Rules 8-10: exit_loop injection into loop children."""

    def _make_mock_llm_agent(self, name: str, tools: list | None = None) -> MagicMock:
        from google.adk.agents import LlmAgent
        agent = MagicMock(spec=LlmAgent)
        agent.name = name
        agent.tools = tools if tools is not None else []
        return agent

    def _make_mock_seq_agent(self, name: str) -> MagicMock:
        from google.adk.agents import SequentialAgent
        agent = MagicMock(spec=SequentialAgent)
        agent.name = name
        agent.tools = []
        return agent

    def test_injects_exit_loop(self):
        """Rule 8: last LlmAgent gets exit_loop."""
        from google.adk.tools.exit_loop_tool import exit_loop

        agent = self._make_mock_llm_agent("a", tools=[])
        _inject_exit_loop([agent])
        assert exit_loop in agent.tools

    def test_skip_if_already_has_exit_loop(self):
        """Rule 9: no duplicate exit_loop."""
        from google.adk.tools.exit_loop_tool import exit_loop

        agent = self._make_mock_llm_agent("a", tools=[exit_loop])
        _inject_exit_loop([agent])
        assert agent.tools.count(exit_loop) == 1

    def test_skips_non_llm_agent(self):
        """Rule 10: SequentialAgent is not modified."""
        seq = self._make_mock_seq_agent("s")
        original_tools = list(seq.tools)
        _inject_exit_loop([seq])
        assert seq.tools == original_tools

    def test_injects_into_last_llm(self):
        """exit_loop goes into the *last* LlmAgent only."""
        from google.adk.tools.exit_loop_tool import exit_loop

        a1 = self._make_mock_llm_agent("first", tools=[])
        a2 = self._make_mock_llm_agent("second", tools=[])
        _inject_exit_loop([a1, a2])
        assert exit_loop in a2.tools
        assert exit_loop not in a1.tools

    def test_none_tools_becomes_list(self):
        """Agent with tools=None gets [exit_loop]."""
        from google.adk.tools.exit_loop_tool import exit_loop

        agent = self._make_mock_llm_agent("a", tools=None)
        _inject_exit_loop([agent])
        assert agent.tools == [exit_loop]


# ---- Rules 11-15: build() ----


class TestBuildSequentialTree:
    """Rule 11: PipelineConfig → SequentialAgent with children."""

    @patch("fedotmas.pipeline.builder.make_callbacks", return_value=(MagicMock(), MagicMock()))
    @patch("fedotmas.pipeline.builder.create_toolset", return_value=[])
    def test_sequential(self, _mock_toolset, _mock_cb, simple_pipeline_config):
        root = build(simple_pipeline_config)
        from google.adk.agents import SequentialAgent
        assert isinstance(root, SequentialAgent)
        assert len(root.sub_agents) == 2
        assert root.sub_agents[0].name == "alpha"
        assert root.sub_agents[1].name == "beta"


class TestBuildParallelTree:
    """Rule 12: parallel type → ParallelAgent."""

    @patch("fedotmas.pipeline.builder.make_callbacks", return_value=(MagicMock(), MagicMock()))
    @patch("fedotmas.pipeline.builder.create_toolset", return_value=[])
    def test_parallel(self, _mock_toolset, _mock_cb):
        config = PipelineConfig.model_validate({
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
        })
        root = build(config)
        from google.adk.agents import ParallelAgent
        assert isinstance(root, ParallelAgent)
        assert len(root.sub_agents) == 2


class TestBuildNestedSeqPar:
    """Rule 13: nested sequential → parallel."""

    @patch("fedotmas.pipeline.builder.make_callbacks", return_value=(MagicMock(), MagicMock()))
    @patch("fedotmas.pipeline.builder.create_toolset", return_value=[])
    def test_nested(self, _mock_toolset, _mock_cb):
        config = PipelineConfig.model_validate({
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
        })
        root = build(config)
        from google.adk.agents import ParallelAgent, SequentialAgent
        assert isinstance(root, SequentialAgent)
        assert isinstance(root.sub_agents[1], ParallelAgent)
        assert len(root.sub_agents[1].sub_agents) == 2


class TestBuildLoopMaxIterations:
    """Rule 14: loop with explicit max_iterations."""

    @patch("fedotmas.pipeline.builder.make_callbacks", return_value=(MagicMock(), MagicMock()))
    @patch("fedotmas.pipeline.builder.create_toolset", return_value=[])
    def test_loop_explicit(self, _mock_toolset, _mock_cb):
        config = PipelineConfig.model_validate({
            "agents": [
                {"name": "worker", "instruction": "iterate", "output_key": "result"},
            ],
            "pipeline": {
                "type": "loop",
                "max_iterations": 5,
                "children": [
                    {"type": "agent", "agent_name": "worker"},
                ],
            },
        })
        root = build(config)
        from google.adk.agents import LoopAgent
        assert isinstance(root, LoopAgent)
        assert root.max_iterations == 5


class TestBuildLoopDefaultMaxIterations:
    """Rule 15: loop without max_iterations → settings default."""

    @patch("fedotmas.pipeline.builder.make_callbacks", return_value=(MagicMock(), MagicMock()))
    @patch("fedotmas.pipeline.builder.create_toolset", return_value=[])
    @patch("fedotmas.pipeline.builder.get_max_loop_iterations", return_value=10)
    def test_loop_default(self, _mock_max, _mock_toolset, _mock_cb):
        config = PipelineConfig.model_validate({
            "agents": [
                {"name": "worker", "instruction": "iterate", "output_key": "result"},
            ],
            "pipeline": {
                "type": "loop",
                "children": [
                    {"type": "agent", "agent_name": "worker"},
                ],
            },
        })
        root = build(config)
        from google.adk.agents import LoopAgent
        assert isinstance(root, LoopAgent)
        assert root.max_iterations == 10
