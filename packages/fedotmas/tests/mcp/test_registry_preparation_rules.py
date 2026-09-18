"""What a run does about MCP servers it cannot describe or cannot reach."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from fedotmas import MAW
from fedotmas.mcp._config import HttpMCPServer, StdioMCPServer
from fedotmas.mcp.describe import UnreachableServer


def _described(registry, name, description):
    from dataclasses import replace

    return {**registry, name: replace(registry[name], description=description)}


class TestSuppliedServers:
    """A registry the caller built itself names addresses of its own."""

    async def test_an_unreachable_supplied_server_stops_generation(self):
        registry = {"remote": HttpMCPServer("https://typo.invalid/mcp")}
        maw = MAW(mcp_servers=registry)

        with patch(
            "fedotmas.core.base.describe_servers",
            return_value=(registry, [UnreachableServer("remote", "timed out")]),
        ):
            with pytest.raises(ValueError, match="did not answer: remote"):
                await maw.generate_config("task")

    async def test_the_failure_names_the_reason(self):
        registry = {"remote": HttpMCPServer("https://typo.invalid/mcp")}
        maw = MAW(mcp_servers=registry)

        with patch(
            "fedotmas.core.base.describe_servers",
            return_value=(
                registry,
                [UnreachableServer("remote", "ConnectionError: refused")],
            ),
        ):
            with pytest.raises(ValueError, match="ConnectionError: refused"):
                await maw.generate_config("task")


class TestDiscoveredServers:
    """A workspace server that is broken is an environment problem, not a typo."""

    async def test_an_unreachable_discovered_server_is_dropped(self, monkeypatch):
        registry = {
            "broken": StdioMCPServer("x", ()),
            "fine": StdioMCPServer("y", (), description="Works."),
        }
        monkeypatch.setattr(
            "fedotmas.mcp.get_mcp_servers", lambda: registry, raising=False
        )
        maw = MAW(mcp_servers=["broken", "fine"])

        with patch(
            "fedotmas.core.base.describe_servers",
            return_value=(registry, [UnreachableServer("broken", "timed out")]),
        ):
            await maw._prepare_mcp_registry()

        assert set(maw.mcp_registry) == {"fine"}


class TestPreparation:
    async def test_descriptions_reach_the_registry(self):
        registry = {"remote": HttpMCPServer("https://example.org/mcp")}
        maw = MAW(mcp_servers=registry)

        with patch(
            "fedotmas.core.base.describe_servers",
            return_value=(_described(registry, "remote", "Tools: search (...)."), []),
        ):
            await maw._prepare_mcp_registry()

        assert maw.mcp_registry["remote"].description == "Tools: search (...)."

    async def test_a_failed_preparation_is_retried(self):
        """The caller may start the server and try again on the same instance."""
        registry = {"remote": HttpMCPServer("https://example.org/mcp")}
        maw = MAW(mcp_servers=registry)

        with patch(
            "fedotmas.core.base.describe_servers",
            return_value=(registry, [UnreachableServer("remote", "timed out")]),
        ):
            with pytest.raises(ValueError):
                await maw._prepare_mcp_registry()

        with patch(
            "fedotmas.core.base.describe_servers",
            return_value=(_described(registry, "remote", "Tools: search (...)."), []),
        ) as probe:
            await maw._prepare_mcp_registry()

        probe.assert_called_once()
        assert maw.mcp_registry["remote"].description == "Tools: search (...)."

    async def test_a_described_server_is_never_contacted(self, monkeypatch):
        """Its reachability is `just doctor`'s business, not every run's."""
        probed = []

        async def listing(name, registry=None, *, timeout=None):
            probed.append(name)
            return []

        monkeypatch.setattr("fedotmas.mcp.describe.list_server_tools", listing)
        maw = MAW(
            mcp_servers={
                "declared": HttpMCPServer("https://example.org/a", description="Does."),
                "bare": HttpMCPServer("https://example.org/b"),
            }
        )

        await maw._prepare_mcp_registry()

        assert probed == ["bare"]
        assert maw.mcp_registry["declared"].description == "Does."

    async def test_it_probes_only_once(self):
        registry = {"remote": HttpMCPServer("https://example.org/mcp")}
        maw = MAW(mcp_servers=registry)

        with patch(
            "fedotmas.core.base.describe_servers", return_value=(registry, [])
        ) as probe:
            await maw._prepare_mcp_registry()
            await maw._prepare_mcp_registry()

        assert probe.call_count == 1

    async def test_a_tool_catalog_skips_probing_entirely(self):
        maw = MAW(
            mcp_servers={"remote": HttpMCPServer("https://typo.invalid/mcp")},
            tool_catalog={"urban.getproject": "Fetch a project"},
        )

        with patch("fedotmas.core.base.describe_servers") as probe:
            await maw._prepare_mcp_registry()

        probe.assert_not_called()

    async def test_an_empty_registry_probes_nothing(self):
        maw = MAW()

        with patch("fedotmas.core.base.describe_servers") as probe:
            await maw._prepare_mcp_registry()

        probe.assert_not_called()
