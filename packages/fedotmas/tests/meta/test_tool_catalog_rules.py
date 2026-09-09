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


def _config(tools: list[str]):
    from fedotmas import MAWConfig
    from fedotmas.maw.models import MAWAgentConfig, MAWStepConfig

    return MAWConfig(
        agents=[
            MAWAgentConfig(
                name="fetcher",
                instruction="Fetch it",
                output_key="fetched",
                tools=tools,
            )
        ],
        pipeline=MAWStepConfig(type="agent", agent_name="fetcher"),
    )


class TestForeignToolsRejectedAtBuild:
    """Rule 4: a config generated against a catalogue is not buildable locally."""

    def test_build_names_the_catalogue(self):
        from fedotmas import MAW

        maw = MAW(tool_catalog={"urban.getproject": "Get a project"})

        with pytest.raises(ValueError, match="tool_catalog"):
            maw.build(_config(["urban.getproject"]))

    def test_a_name_the_registry_also_has_is_still_theirs(self):
        from fedotmas import MAW

        maw = MAW(
            mcp_servers=_REGISTRY, tool_catalog={"download": "Their own downloader"}
        )

        with pytest.raises(ValueError, match="tool_catalog"):
            maw.build(_config(["download"]))

    def test_a_toolless_config_is_refused_too(self):
        from fedotmas import MAW

        maw = MAW(tool_catalog={"urban.getproject": "Get a project"})

        with pytest.raises(ValueError, match="tool_catalog"):
            maw.build(_config([]))

    def test_an_instance_without_a_catalogue_builds(self):
        from fedotmas import MAW

        MAW().build(_config([]))


class TestGeneratedIdsAreStripped:
    """Rule 5: `id` is the caller's to set, even though the pool schema exposes it."""

    async def test_pool_stage_clears_invented_ids(self):
        from fedotmas._settings import ModelConfig
        from fedotmas.meta._adk_runner import LLMCallResult
        from fedotmas.meta.maw_pool_stage import PoolGenerator

        async def _pool(**kwargs):
            return LLMCallResult(
                raw_output={
                    "agents": [
                        {"name": "a", "instruction": "x", "id": "somebodys_record"}
                    ]
                },
                prompt_tokens=1,
                completion_tokens=1,
                elapsed=0.1,
            )

        with patch(
            "fedotmas.meta.maw_pool_stage.run_meta_agent_call", side_effect=_pool
        ):
            gen = PoolGenerator(
                meta_model=ModelConfig(model="openai/gpt-4o"),
                worker_models=[ModelConfig(model="openai/gpt-4o")],
                tool_catalog={},
            )
            pool = await gen.generate("task")

        assert pool.agents[0].id is None
