"""Tool catalogue rules — what the meta-agent is told it may call."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from fedotmas.mcp import StdioMCPServer
from fedotmas.meta._helpers import resolve_tool_descriptions


_REGISTRY = {
    "download": StdioMCPServer(
        command="uv", args=("run", "server"), description="Fetch files"
    )
}


class TestExternalCatalogueWins:
    """Rule 1: an explicit catalogue replaces the local MCP registry."""

    def test_catalogue_replaces_registry(self):
        result = resolve_tool_descriptions(
            _REGISTRY, {"urban.getproject": "Get a project"}
        )
        assert result == {"urban.getproject": "Get a project"}


class TestEmptyCatalogueMeansNoTools:
    """Rule 2: an empty catalogue is 'no tools', not 'fall back to the registry'."""

    def test_empty_catalogue_advertises_nothing(self):
        assert resolve_tool_descriptions(_REGISTRY, {}) == {}


class TestNoCatalogueFallsBack:
    """Rule 3: without a catalogue the local registry is described as before."""

    def test_registry_used(self):
        assert resolve_tool_descriptions(_REGISTRY, None) == {"download": "Fetch files"}

    def test_discovery_used_when_registry_absent(self):
        with patch(
            "fedotmas.meta._helpers.get_server_descriptions", return_value={"a": "b"}
        ) as mocked:
            assert resolve_tool_descriptions(None, None) == {"a": "b"}
        mocked.assert_called_once_with(None)


class TestForeignToolsRejectedAtBuild:
    """Rule 4: a config generated against a catalogue is not buildable locally."""

    def test_build_names_the_catalogue(self):
        from fedotmas import MAW, MAWConfig
        from fedotmas.maw.models import MAWAgentConfig, MAWStepConfig

        config = MAWConfig(
            agents=[
                MAWAgentConfig(
                    name="fetcher",
                    instruction="Fetch it",
                    output_key="fetched",
                    tools=["urban.getproject"],
                )
            ],
            pipeline=MAWStepConfig(type="agent", agent_name="fetcher"),
        )
        maw = MAW(tool_catalog={"urban.getproject": "Get a project"})

        with pytest.raises(ValueError, match="tool_catalog"):
            maw.build(config)
