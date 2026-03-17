from __future__ import annotations

import os

from fastmcp.client.transports import StdioTransport

from mcp_web_scraping.server import mcp, transport


class TestProxyConfig:
    def test_server_name(self):
        assert mcp.name == "web-scraping"

    def test_transport_type(self):
        assert isinstance(transport, StdioTransport)

    def test_transport_command(self):
        assert transport.command == "lightpanda"

    def test_transport_args(self):
        assert transport.args == ["mcp", "--insecure_disable_tls_host_verification"]

    def test_env_inherits_process_env(self):
        assert transport.env is not None
        assert transport.env["HOME"] == os.environ["HOME"]

    def test_env_would_include_api_keys(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-456")
        from importlib import reload

        import mcp_web_scraping.server as mod

        reload(mod)
        assert mod.transport.env["OPENAI_API_KEY"] == "test-key-456"
