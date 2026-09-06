"""Builder edge-case tests for MAS routing system — patch ADK constructors."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from fedotmas.mas.builder import build_routing_system
from fedotmas.mas.models import MASConfig
from fedotmas.maw.builder import AUTONOMY_CLOSING, AUTONOMY_PREAMBLE
from google.adk.agents import LlmAgent
from google.adk.tools.base_toolset import BaseToolset


def _config(**overrides) -> MASConfig:
    """Helper to build a minimal MASConfig with overrides."""
    base = {
        "coordinator": {
            "name": "coord",
            "description": "Routes tasks",
            "instruction": "Route requests.",
        },
        "workers": [
            {
                "name": "alpha",
                "description": "Does alpha work",
                "instruction": "Do alpha.",
                "output_key": "alpha_out",
            },
            {
                "name": "beta",
                "description": "Does beta work",
                "instruction": "Do beta.",
                "output_key": "beta_out",
            },
        ],
    }
    base.update(overrides)
    return MASConfig.model_validate(base)


class TestBuildFlatHierarchy:
    """Rule 1: Coordinator gets workers as sub_agents (flat, not nested)."""

    @patch("fedotmas.mas.builder.create_toolset", return_value=[])
    def test_flat_hierarchy(self, _mock_toolset):
        config = _config()
        root = build_routing_system(config)
        assert isinstance(root, LlmAgent)
        assert len(root.sub_agents) == 2
        assert root.sub_agents[0].name == "alpha"
        assert root.sub_agents[1].name == "beta"


class TestWorkerDescriptionPassedThrough:
    """Rule 2: Each LlmAgent receives description= kwarg."""

    @patch("fedotmas.mas.builder.create_toolset", return_value=[])
    def test_worker_descriptions(self, _mock_toolset):
        config = _config()
        root = build_routing_system(config)
        assert root.sub_agents[0].description == "Does alpha work"
        assert root.sub_agents[1].description == "Does beta work"


class TestCoordinatorHasDescription:
    """Rule 3: Coordinator's description is passed through."""

    @patch("fedotmas.mas.builder.create_toolset", return_value=[])
    def test_coordinator_description(self, _mock_toolset):
        config = _config()
        root = build_routing_system(config)
        assert root.description == "Routes tasks"


class TestMCPToolResolution:
    """Rule 4: Workers with tools → create_toolset called per tool."""

    @patch("fedotmas.mas.builder.create_toolset")
    def test_tools_resolved(self, mock_toolset):
        mock_toolset.return_value = MagicMock(spec=BaseToolset)
        config = MASConfig(
            coordinator={
                "name": "coord",
                "description": "Routes",
                "instruction": "Route.",
            },
            workers=[
                {
                    "name": "coder",
                    "description": "Writes code",
                    "instruction": "Code.",
                    "tools": ["sandbox", "web-search"],
                },
            ],
        )
        build_routing_system(config)
        assert mock_toolset.call_count == 2


class TestNoToolsAgent:
    """Rule 5: Worker with empty tools → LlmAgent with tools=[]."""

    @patch("fedotmas.mas.builder.create_toolset", return_value=[])
    def test_empty_tools(self, mock_toolset):
        config = _config()
        root = build_routing_system(config)
        for worker in root.sub_agents:
            assert worker.tools == []
        # create_toolset should not be called for agents without tools
        mock_toolset.assert_not_called()


class TestModelResolution:
    """Rule 6: Worker model resolved through _resolve_llm."""

    @patch("fedotmas.mas.builder.create_toolset", return_value=[])
    @patch("fedotmas.mas.builder._resolve_llm")
    def test_custom_model(self, mock_resolve, _mock_toolset):
        mock_resolve.return_value = "resolved-model"
        config = _config()
        build_routing_system(config)
        # _resolve_llm called for coordinator + 2 workers = 3 times
        assert mock_resolve.call_count == 3


class TestOutputKeyForwarded:
    """Rule 7: output_key from config → LlmAgent(output_key=...)."""

    @patch("fedotmas.mas.builder.create_toolset", return_value=[])
    def test_output_key(self, _mock_toolset):
        config = _config()
        root = build_routing_system(config)
        assert root.sub_agents[0].output_key == "alpha_out"
        assert root.sub_agents[1].output_key == "beta_out"


class TestAutonomyPreamble:
    """Routing runs unattended too: a coordinator must not stall on a reply."""

    def test_every_agent_in_the_tree_carries_it(self):
        coord = build_routing_system(_config())

        for agent in [coord, *coord.sub_agents]:
            assert AUTONOMY_PREAMBLE in agent.instruction
            assert agent.instruction.endswith(AUTONOMY_CLOSING)

    def test_each_agents_own_instruction_survives_intact(self):
        coord = build_routing_system(_config())

        assert "Route requests." in coord.instruction
        middles = [
            a.instruction.split("\n\n", 1)[1].rsplit("\n\n", 1)[0]
            for a in coord.sub_agents
        ]
        assert middles == ["Do alpha.", "Do beta."]

    def test_a_caller_with_a_person_in_the_loop_can_turn_it_off(self):
        coord = build_routing_system(_config(), autonomous=False)

        for agent in [coord, *coord.sub_agents]:
            assert AUTONOMY_PREAMBLE not in agent.instruction
            assert AUTONOMY_CLOSING not in agent.instruction
        assert coord.instruction == "Route requests."


class TestSingleTurnWorkerRegistration:
    """Workers are registered as call-and-return coordinator tools."""

    @patch("fedotmas.mas.builder.create_toolset", return_value=[])
    def test_builds_workers_for_sequential_delegation(self, _mock_toolset):
        config = MASConfig(
            coordinator={
                "name": "coordinator",
                "description": "Coordinates Fibonacci calculation and verification",
                "instruction": (
                    "Call fib_calculator with N=20, pass its returned sequence "
                    "to fib_verifier, then give the final answer."
                ),
            },
            workers=[
                {
                    "name": "fib_calculator",
                    "description": "Calculates Fibonacci sequences",
                    "instruction": "Return exactly the requested Fibonacci sequence.",
                },
                {
                    "name": "fib_verifier",
                    "description": "Verifies Fibonacci sequences",
                    "instruction": "Validate the supplied sequence.",
                },
            ],
        )

        coordinator = build_routing_system(config)

        # ADK exposes single-turn children as callable tools. This permits the
        # required control flow: coordinator -> calculator(N=20) ->
        # coordinator -> verifier(sequence) -> coordinator -> final answer.
        assert coordinator.mode is None
        assert [worker.mode for worker in coordinator.sub_agents] == [
            "single_turn",
            "single_turn",
        ]
        assert [worker.name for worker in coordinator.sub_agents] == [
            "fib_calculator",
            "fib_verifier",
        ]

    async def test_exposes_single_turn_workers_as_coordinator_tools(self):
        coordinator = build_routing_system(_config(), autonomous=False)

        tools = await coordinator.canonical_tools()
        assert [tool.name for tool in tools] == ["alpha", "beta"]
        assert all(
            worker.disallow_transfer_to_parent for worker in coordinator.sub_agents
        )

    @patch("fedotmas.mas.builder.create_toolset", return_value=[])
    @patch("fedotmas.mas.builder.LlmAgent", wraps=LlmAgent)
    def test_registers_single_turn_workers_during_coordinator_construction(
        self, agent_constructor, _mock_toolset
    ):
        build_routing_system(_config())

        coordinator_call = next(
            call
            for call in agent_constructor.call_args_list
            if call.kwargs["name"] == "coord"
        )
        workers = coordinator_call.kwargs["sub_agents"]
        assert [worker.name for worker in workers] == ["alpha", "beta"]
        assert [worker.mode for worker in workers] == ["single_turn", "single_turn"]
