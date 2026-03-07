"""Builder edge-case tests — patch ADK constructors, test logic."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from fedotmas._settings import ModelConfig
from fedotmas.maw.builder import (
    _inject_exit_loop,
    _resolve_llm,
    build,
)
from fedotmas.maw.models import MAWAgentConfig, MAWConfig


# ---- Rules 1-3: text normalization (via MAWAgentConfig model_validator) ----


class TestNormalizeAngleBrackets:
    """Rule 1: <var> → {var} (via MAWAgentConfig)."""

    def _make(self, instruction: str) -> MAWAgentConfig:
        return MAWAgentConfig(name="t", instruction=instruction, output_key="k")

    def test_single_var(self):
        assert self._make("<var>").instruction == "{var?}"

    def test_multiple_vars(self):
        assert self._make("<a> and <b>").instruction == "{a?} and {b?}"

    def test_no_vars(self):
        assert self._make("plain text").instruction == "plain text"


class TestMakeVarsOptional:
    """Rule 2: {var} → {var?} (via MAWAgentConfig)."""

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
    """Rule 3: <var> → {var} → {var?} chain (via MAWAgentConfig)."""

    def test_chain(self):
        cfg = MAWAgentConfig(
            name="t",
            instruction="Process <input> and <context>",
            output_key="k",
        )
        assert cfg.instruction == "Process {input?} and {context?}"


# ---- Rules 4-5: model normalization (via MAWAgentConfig) ----


class TestNormalizeModelNameDefault:
    """Rule 4: None/empty model → stays None (resolved later by _resolve_llm)."""

    def test_none_model(self):
        cfg = MAWAgentConfig(name="t", instruction="x", output_key="k", model=None)
        assert cfg.model is None

    def test_resolve_none_gives_default(self):
        result = _resolve_llm(None, None)
        assert result == "openai/gpt-oss-120b"


class TestNormalizeModelNamePrefix:
    """Rule 5: bare model name gets openai/ prefix (via MAWAgentConfig)."""

    def test_bare_name(self):
        cfg = MAWAgentConfig(name="t", instruction="x", output_key="k", model="gpt-4o")
        assert cfg.model == "openai/gpt-4o"

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


# ---- Rules 6-7: _resolve_llm ----


class TestResolveLlmCustomEndpoint:
    """Rule 6: model in worker_models → delegates to make_llm factory."""

    @patch("fedotmas.maw.builder.make_llm")
    def test_custom_endpoint(self, mock_factory):
        cfg = ModelConfig(model="my-model", api_base="http://localhost:9090")
        _resolve_llm("my-model", {"my-model": cfg})
        mock_factory.assert_called_once_with(cfg)

    @patch("fedotmas.maw.builder.make_llm")
    def test_custom_with_api_key(self, mock_factory):
        cfg = ModelConfig(model="m", api_base="http://x", api_key="sk-123")
        _resolve_llm("m", {"m": cfg})
        mock_factory.assert_called_once_with(cfg)


class TestResolveLlmNoCustom:
    """Rule 7: model not in worker_models → plain string."""

    def test_not_in_registry(self):
        other = ModelConfig(model="other")
        result = _resolve_llm("openai/gpt-4o", {"other": other})
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
    """Rule 11: MAWConfig → SequentialAgent with children."""

    @patch("fedotmas.maw.builder.create_toolset", return_value=[])
    def test_sequential(self, _mock_toolset, simple_pipeline_config):
        root = build(simple_pipeline_config)
        from google.adk.agents import SequentialAgent

        assert isinstance(root, SequentialAgent)
        assert len(root.sub_agents) == 2
        assert root.sub_agents[0].name == "alpha"
        assert root.sub_agents[1].name == "beta"


class TestBuildParallelTree:
    """Rule 12: parallel type → ParallelAgent."""

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
        from google.adk.agents import ParallelAgent

        assert isinstance(root, ParallelAgent)
        assert len(root.sub_agents) == 2


class TestBuildNestedSeqPar:
    """Rule 13: nested sequential → parallel."""

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
        from google.adk.agents import ParallelAgent, SequentialAgent

        assert isinstance(root, SequentialAgent)
        assert isinstance(root.sub_agents[1], ParallelAgent)
        assert len(root.sub_agents[1].sub_agents) == 2


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
        from google.adk.agents import LoopAgent

        assert isinstance(root, LoopAgent)
        assert root.max_iterations == 5


class TestBuildLoopDefaultMaxIterations:
    """Rule 15: loop without max_iterations → settings default."""

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
        from google.adk.agents import LoopAgent

        assert isinstance(root, LoopAgent)
        assert root.max_iterations == 10
