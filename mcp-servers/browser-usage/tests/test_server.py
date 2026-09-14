from __future__ import annotations

import asyncio
import base64
import json
import os
import sys

import mcp.types as mt
from fastmcp.client.transports import StdioTransport
from fastmcp.server.middleware import MiddlewareContext
from fastmcp.tools import ToolResult

from mcp_browser_usage.server import ScreenshotToFile, mcp, transport

PNG = b"\x89PNG\r\n\x1a\nfake"


def _run(middleware, name, text, is_error=False):
    context = MiddlewareContext(
        message=mt.CallToolRequestParams(name=name, arguments={})
    )
    result = ToolResult(
        content=[mt.TextContent(type="text", text=text)], is_error=is_error
    )

    async def call_next(_):
        return result

    return asyncio.run(middleware.on_call_tool(context, call_next))


def _payload(result):
    return json.loads(result.content[0].text)


class TestScreenshotToFile:
    def test_screenshot_is_replaced_by_a_file(self, tmp_path):
        text = json.dumps({"screenshot": base64.b64encode(PNG).decode(), "size_bytes": 12})
        result = _run(ScreenshotToFile(tmp_path), "browser_screenshot", text)

        payload = _payload(result)
        assert "screenshot" not in payload
        assert payload["size_bytes"] == 12
        path = payload["screenshot_path"]
        assert path.endswith(".png")
        with open(path, "rb") as f:
            assert f.read() == PNG

    def test_get_state_keeps_the_page_fields(self, tmp_path):
        text = json.dumps(
            {"url": "https://example.org", "screenshot": base64.b64encode(PNG).decode()}
        )
        payload = _payload(_run(ScreenshotToFile(tmp_path), "browser_get_state", text))

        assert payload["url"] == "https://example.org"
        assert "screenshot_path" in payload

    def test_state_without_screenshot_is_untouched(self, tmp_path):
        text = json.dumps({"url": "https://example.org"})
        result = _run(ScreenshotToFile(tmp_path), "browser_get_state", text)

        assert result.content[0].text == text
        assert list(tmp_path.iterdir()) == []

    def test_non_json_text_is_untouched(self, tmp_path):
        text = "Error: No browser session active"
        result = _run(ScreenshotToFile(tmp_path), "browser_screenshot", text)

        assert result.content[0].text == text

    def test_other_tools_are_untouched(self, tmp_path):
        text = json.dumps({"screenshot": base64.b64encode(PNG).decode()})
        result = _run(ScreenshotToFile(tmp_path), "browser_extract_content", text)

        assert result.content[0].text == text

    def test_undecodable_screenshot_is_dropped(self, tmp_path):
        text = json.dumps({"screenshot": "not base64!"})
        payload = _payload(_run(ScreenshotToFile(tmp_path), "browser_screenshot", text))

        assert "screenshot" not in payload
        assert "screenshot_error" in payload


class TestProxyConfig:
    def test_server_name(self):
        assert mcp.name == "browser-usage"

    def test_transport_type(self):
        assert isinstance(transport, StdioTransport)

    def test_transport_runs_upstream_through_the_guard(self):
        assert transport.command == sys.executable
        assert transport.args == [
            "-m",
            "mcp_browser_usage._guard",
            # The proxy's own pid: this module runs inside the proxy.
            str(os.getpid()),
            "uvx",
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
