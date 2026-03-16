from __future__ import annotations

from fedotmas.mcp._config import StdioMCPServer
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
