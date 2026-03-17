from __future__ import annotations

import os

from fastmcp.client.transports import StdioTransport

from mcp_browser_usage.server import mcp, transport


class TestProxyConfig:
    def test_server_name(self):
        assert mcp.name == "browser-usage"

    def test_transport_type(self):
        assert isinstance(transport, StdioTransport)

    def test_transport_command(self):
        assert transport.command == "uvx"

    def test_transport_args(self):
        assert transport.args == [
            "--from",
            "browser-use[cli]",
            "browser-use",
            "--mcp",
        ]

    def test_env_inherits_process_env(self):
        assert transport.env is not None
        assert transport.env["HOME"] == os.environ["HOME"]

    def test_env_would_include_api_keys(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-123")
        from importlib import reload

        import mcp_browser_usage.server as mod

        reload(mod)
        assert mod.transport.env["OPENAI_API_KEY"] == "test-key-123"

    def test_headless_by_default(self, monkeypatch):
        monkeypatch.delenv("BROWSER_USE_HEADLESS", raising=False)
        from importlib import reload

        import mcp_browser_usage.server as mod

        reload(mod)
        assert mod.transport.env["BROWSER_USE_HEADLESS"] == "true"

    def test_headless_respects_explicit_env(self, monkeypatch):
        monkeypatch.setenv("BROWSER_USE_HEADLESS", "false")
        from importlib import reload

        import mcp_browser_usage.server as mod

        reload(mod)
        assert mod.transport.env["BROWSER_USE_HEADLESS"] == "false"

    def test_llm_model_default(self, monkeypatch):
        monkeypatch.delenv("BROWSER_USE_LLM_MODEL", raising=False)
        from importlib import reload

        import mcp_browser_usage.server as mod

        reload(mod)
        assert mod.transport.env["BROWSER_USE_LLM_MODEL"] == "openai/gpt-4o-mini"

    def test_llm_model_respects_explicit_env(self, monkeypatch):
        monkeypatch.setenv("BROWSER_USE_LLM_MODEL", "anthropic/claude-sonnet-4-20250514")
        from importlib import reload

        import mcp_browser_usage.server as mod

        reload(mod)
        assert mod.transport.env["BROWSER_USE_LLM_MODEL"] == "anthropic/claude-sonnet-4-20250514"
