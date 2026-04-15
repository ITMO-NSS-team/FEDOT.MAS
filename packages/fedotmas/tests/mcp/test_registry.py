from __future__ import annotations

from fedotmas.interfaces.tools import ToolDescriptor
from fedotmas.mcp._config import StdioMCPServer
from fedotmas.mcp.registry import create_toolset


class TestCreateToolsetReturnsDescriptor:
    def test_returns_tool_descriptor(self):
        cfg = StdioMCPServer(command="echo", args=("hello",))
        registry = {"dummy": cfg}
        toolset = create_toolset("dummy", registry=registry)

        assert isinstance(toolset, ToolDescriptor)
        assert toolset.name == "dummy"
        assert toolset.mcp_server_name == "dummy"
        assert toolset.mcp_server is cfg


class TestStdioEnvPropagation:
    def test_inherits_parent_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")
        monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:9090/v1")

        cfg = StdioMCPServer(command="echo", args=("hello",))
        registry = {"dummy": cfg}
        toolset = create_toolset("dummy", registry=registry)

        # Build the ADK toolset to verify env propagation
        from fedotmas.backends.adk.builder import _build_mcp_toolset

        mcp_toolset = _build_mcp_toolset(toolset)
        env = mcp_toolset._connection_params.server_params.env
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

        from fedotmas.backends.adk.builder import _build_mcp_toolset

        mcp_toolset = _build_mcp_toolset(toolset)
        env = mcp_toolset._connection_params.server_params.env
        assert env["OPENAI_API_KEY"] == "sk-override"

    def test_includes_default_env_keys(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        cfg = StdioMCPServer(command="echo", args=())
        registry = {"dummy": cfg}
        toolset = create_toolset("dummy", registry=registry)

        from fedotmas.backends.adk.builder import _build_mcp_toolset

        mcp_toolset = _build_mcp_toolset(toolset)
        env = mcp_toolset._connection_params.server_params.env
        assert "PATH" in env
        assert "HOME" in env
