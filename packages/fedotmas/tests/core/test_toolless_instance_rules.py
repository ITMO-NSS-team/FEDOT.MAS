"""Toolless instance rules — an instance with no tools must not be built in silence."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from fedotmas import MAW
from fedotmas.plugins import LoggingPlugin, WebSearchLimitPlugin


def _warnings(**kwargs) -> list[str]:
    # setup_logging() drops every loguru sink, so patch the logger instead.
    with patch("fedotmas.core.base._log") as log:
        MAW(**kwargs)
    return [call.args[0] for call in log.warning.call_args_list]


class TestMissingToolsWarn:
    """Rule 1: an empty registry is reported, since agents then get no tools."""

    def test_servers_left_out(self):
        assert any("No MCP servers" in message for message in _warnings())

    def test_catalogue_that_yields_nothing(self):
        """'all' is no guarantee: discovery can come back empty."""
        with patch("fedotmas.core.base.resolve_mcp_registry", return_value={}):
            messages = _warnings(mcp_servers="all")
        assert any("No MCP servers" in message for message in messages)


class TestDeliberateChoicesStayQuiet:
    """Rule 2: an explicit empty list or an external catalogue is not a mistake."""

    def test_explicit_empty_list(self):
        assert _warnings(mcp_servers=[]) == []

    def test_tool_catalogue(self):
        assert _warnings(tool_catalog={"urban.getproject": "Get a project"}) == []


class TestSearchBudgetNeedsTheDefaultPlugins:
    """Rule 3: web_search_limit configures a plugin set that plugins= replaces."""

    def test_limit_alongside_plugins_is_refused(self):
        with pytest.raises(ValueError, match="web_search_limit"):
            MAW(mcp_servers=[], plugins=[LoggingPlugin()], web_search_limit=20)

    def test_disabling_alongside_plugins_is_refused_too(self):
        with pytest.raises(ValueError, match="web_search_limit"):
            MAW(mcp_servers=[], plugins=[LoggingPlugin()], web_search_limit=None)

    def test_own_plugins_are_kept_as_given(self):
        plugin = LoggingPlugin()
        maw = MAW(mcp_servers=[], plugins=[plugin])
        assert maw._plugins == [plugin]

    def test_limit_reaches_the_default_set(self):
        maw = MAW(mcp_servers=[], web_search_limit=7)
        budget = next(p for p in maw._plugins if isinstance(p, WebSearchLimitPlugin))
        assert budget.max_calls_per_agent == 7

    def test_none_drops_the_budget_from_the_default_set(self):
        maw = MAW(mcp_servers=[], web_search_limit=None)
        assert not any(isinstance(p, WebSearchLimitPlugin) for p in maw._plugins)
