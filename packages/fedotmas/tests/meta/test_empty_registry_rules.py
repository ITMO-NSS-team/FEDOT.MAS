"""Empty registry rules — a toolless instance must not be built in silence."""

from __future__ import annotations

from unittest.mock import patch

from fedotmas import MAW


def _warnings(**kwargs) -> list[str]:
    # setup_logging() drops every loguru sink, so patch the logger instead.
    with patch("fedotmas.core.base._log") as log:
        MAW(**kwargs)
    return [call.args[0] for call in log.warning.call_args_list]


class TestMissingServersWarn:
    """Rule 1: leaving mcp_servers out is reported, since agents get no tools."""

    def test_default_warns(self):
        (message,) = _warnings()
        assert "No MCP servers" in message


class TestDeliberateChoicesStayQuiet:
    """Rule 2: an explicit empty list or an external catalogue is not a mistake."""

    def test_explicit_empty_list(self):
        assert _warnings(mcp_servers=[]) == []

    def test_tool_catalogue(self):
        assert _warnings(tool_catalog={"urban.getproject": "Get a project"}) == []
