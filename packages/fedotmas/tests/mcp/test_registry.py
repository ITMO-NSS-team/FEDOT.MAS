from __future__ import annotations

from fedotmas.mcp._config import (
    DEFAULT_MCP_TIMEOUT_S,
    HttpMCPServer,
    StdioMCPServer,
)
from fedotmas.mcp.registry import create_toolset


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
