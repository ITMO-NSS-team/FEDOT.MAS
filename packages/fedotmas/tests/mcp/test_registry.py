from __future__ import annotations

from pathlib import Path

import pytest

from fedotmas.mcp._config import (
    DEFAULT_MCP_TIMEOUT_S,
    HttpMCPServer,
    StdioMCPServer,
)
from fedotmas.mcp.registry import (
    create_toolset,
    list_server_tools,
    strip_tool_name_prefix,
)
from fedotmas.mcp.discovery import discover_local_servers


class TestStdioEnvPropagation:
    def test_inherits_parent_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")
        monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:9090/v1")

        cfg = StdioMCPServer(command="echo", args=("hello",))
        registry = {"dummy": cfg}
        toolset = create_toolset("dummy", registry=registry)

        env = toolset._connection_params.server_params.env
        assert env["OPENAI_API_KEY"] == "sk-test-123"
        assert env["OPENAI_BASE_URL"] == "http://localhost:9090/v1"

    def test_cfg_env_overrides_parent(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-parent")

        cfg = StdioMCPServer(
            command="echo",
            args=("hello",),
            env={"OPENAI_API_KEY": "sk-override"},
        )
        registry = {"dummy": cfg}
        toolset = create_toolset("dummy", registry=registry)

        env = toolset._connection_params.server_params.env
        assert env["OPENAI_API_KEY"] == "sk-override"

    def test_includes_default_env_keys(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        cfg = StdioMCPServer(command="echo", args=())
        registry = {"dummy": cfg}
        toolset = create_toolset("dummy", registry=registry)

        env = toolset._connection_params.server_params.env
        assert "PATH" in env
        assert "HOME" in env


class TestStdioEnvIsolation:
    """The child resolves its own venv, so ours must not leak into it."""

    def test_parent_virtualenv_is_dropped(self, monkeypatch):
        monkeypatch.setenv("VIRTUAL_ENV", "/repo/.venv")
        monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", "/repo/.venv")

        cfg = StdioMCPServer(command="echo", args=())
        toolset = create_toolset("dummy", registry={"dummy": cfg})

        env = toolset._connection_params.server_params.env
        assert "VIRTUAL_ENV" not in env
        assert "UV_PROJECT_ENVIRONMENT" not in env

    def test_cfg_env_may_still_set_virtualenv(self, monkeypatch):
        monkeypatch.setenv("VIRTUAL_ENV", "/repo/.venv")

        cfg = StdioMCPServer(
            command="echo", args=(), env={"VIRTUAL_ENV": "/elsewhere/.venv"}
        )
        toolset = create_toolset("dummy", registry={"dummy": cfg})

        env = toolset._connection_params.server_params.env
        assert env["VIRTUAL_ENV"] == "/elsewhere/.venv"


class TestTimeoutDefaults:
    """The generous cold-start budget belongs to locally spawned servers only."""

    def test_stdio_gets_the_cold_start_budget(self):
        assert StdioMCPServer(command="echo", args=()).timeout == DEFAULT_MCP_TIMEOUT_S

    def test_http_stays_tight(self):
        """ADK spends this per request, so an unreachable host must fail fast."""
        assert HttpMCPServer(url="http://localhost:9001/mcp").timeout == 60


class TestToolNamePrefixReachesAdk:
    def test_prefix_is_passed_through(self):
        cfg = StdioMCPServer(command="echo", args=(), tool_name_prefix="web_scraping")
        toolset = create_toolset("dummy", registry={"dummy": cfg})

        assert toolset.tool_name_prefix == "web_scraping"

    def test_no_prefix_by_default(self):
        cfg = StdioMCPServer(command="echo", args=())
        toolset = create_toolset("dummy", registry={"dummy": cfg})

        assert toolset.tool_name_prefix is None


class TestStripToolNamePrefix:
    """Policies match tool names literally, so they must see through a prefix."""

    def test_strips_a_registered_prefix(self):
        assert strip_tool_name_prefix("web_scraping_goto") == "goto"

    def test_leaves_an_unprefixed_name_alone(self):
        assert strip_tool_name_prefix("goto") == "goto"

    def test_does_not_strip_a_coincidental_lookalike(self):
        """No registered prefix matches, so the name survives intact."""
        assert strip_tool_name_prefix("download_file") == "download_file"


@pytest.mark.asyncio
async def test_local_code_agent_can_restart_and_exposes_solver_each_time():
    repo = Path(__file__).resolve().parents[4]
    registry = discover_local_servers(repo / "mcp-servers")
    for _ in range(2):
        tools = await list_server_tools("code-agent", registry)
        assert "solve_with_code" in {tool.name for tool in tools}


@pytest.mark.asyncio
async def test_required_local_mcp_startup_failure_is_raised():
    registry = {
        "required": StdioMCPServer(command="/missing/mcp-server", args=(), timeout=1)
    }
    with pytest.raises(Exception):
        await list_server_tools("required", registry)
