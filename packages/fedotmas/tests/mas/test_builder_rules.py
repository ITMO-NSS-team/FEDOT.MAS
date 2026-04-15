"""Builder edge-case tests for MAS routing system — descriptors."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from fedotmas.interfaces.agent import AgentDescriptor, RoutingDescriptor
from fedotmas.mas.builder import build_routing_system
from fedotmas.mas.models import MASConfig


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
        assert isinstance(root, RoutingDescriptor)
        assert len(root.workers) == 2
        assert root.workers[0].name == "alpha"
        assert root.workers[1].name == "beta"


class TestWorkerDescriptionPassedThrough:
    """Rule 2: Each worker receives description."""

    @patch("fedotmas.mas.builder.create_toolset", return_value=[])
    def test_worker_descriptions(self, _mock_toolset):
        config = _config()
        root = build_routing_system(config)
        assert root.workers[0].description == "Does alpha work"
        assert root.workers[1].description == "Does beta work"


class TestCoordinatorHasDescription:
    """Rule 3: Coordinator's description is passed through."""

    @patch("fedotmas.mas.builder.create_toolset", return_value=[])
    def test_coordinator_description(self, _mock_toolset):
        config = _config()
        root = build_routing_system(config)
        assert root.coordinator.description == "Routes tasks"


class TestMCPToolResolution:
    """Rule 4: Workers with tools → create_toolset called per tool."""

    @patch("fedotmas.mas.builder.create_toolset")
    def test_tools_resolved(self, mock_toolset):
        mock_toolset.return_value = MagicMock()
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
    """Rule 5: Worker with empty tools → descriptor with tools=[]."""

    @patch("fedotmas.mas.builder.create_toolset", return_value=[])
    def test_empty_tools(self, mock_toolset):
        config = _config()
        root = build_routing_system(config)
        for worker in root.workers:
            assert worker.tools == []
        # create_toolset should not be called for agents without tools
        mock_toolset.assert_not_called()


class TestModelResolution:
    """Rule 6: Worker model resolved through _resolve_model."""

    @patch("fedotmas.mas.builder.create_toolset", return_value=[])
    @patch("fedotmas.mas.builder._resolve_model")
    def test_custom_model(self, mock_resolve, _mock_toolset):
        mock_resolve.return_value = "resolved-model"
        config = _config()
        root = build_routing_system(config)
        # _resolve_model called for coordinator + 2 workers = 3 times
        assert mock_resolve.call_count == 3


class TestOutputKeyForwarded:
    """Rule 7: output_key from config → descriptor output_key."""

    @patch("fedotmas.mas.builder.create_toolset", return_value=[])
    def test_output_key(self, _mock_toolset):
        config = _config()
        root = build_routing_system(config)
        assert root.workers[0].output_key == "alpha_out"
        assert root.workers[1].output_key == "beta_out"
