from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace

import mcp.types as mt
from fastmcp.client.transports import StdioTransport
from fastmcp.server.middleware import MiddlewareContext
from fastmcp.tools.tool import ToolResult

from mcp_web_scraping.server import ExtractMarkdownFallback, mcp, transport


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


class _FakeServer:
    """Stands in for the proxy: records `markdown` calls and replays answers."""

    def __init__(self, answers):
        self.answers = answers
        self.calls = []

    async def call_tool(self, name, arguments, run_middleware=True):
        self.calls.append((name, arguments, run_middleware))
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


def _context(name, arguments=None, server=None):
    return MiddlewareContext(
        message=mt.CallToolRequestParams(name=name, arguments=arguments or {}),
        fastmcp_context=SimpleNamespace(fastmcp=server),
    )


def _text(*parts):
    return [mt.TextContent(type="text", text=part) for part in parts]


def _run(middleware, context, result):
    async def call_next(_):
        return result

    return asyncio.run(middleware.on_call_tool(context, call_next))


class TestExtractMarkdownFallback:
    def test_successful_extract_is_untouched(self):
        server = _FakeServer([])
        ok = ToolResult(content=_text("extracted"))
        out = _run(
            ExtractMarkdownFallback(), _context("extract", server=server), ok
        )
        assert out is ok
        assert server.calls == []

    def test_other_failing_tools_are_untouched(self):
        server = _FakeServer([])
        failure = ToolResult(content=_text("boom"), is_error=True)
        out = _run(ExtractMarkdownFallback(), _context("links", server=server), failure)
        assert out is failure
        assert server.calls == []

    def test_failed_extract_falls_back_to_markdown(self):
        server = _FakeServer([ToolResult(content=_text("# Page"))])
        failure = ToolResult(
            content=_text("; exception: Cannot read properties of null"),
            is_error=True,
        )
        out = _run(ExtractMarkdownFallback(), _context("extract", server=server), failure)

        assert server.calls == [("markdown", {}, False)]
        assert out.is_error
        assert "Cannot read properties of null" in out.content[0].text
        assert out.content[1].text == "# Page"

    def test_fallback_retries_with_the_last_goto_url(self):
        middleware = ExtractMarkdownFallback()
        server = _FakeServer(
            [
                ToolResult(content=_text("needs a url"), is_error=True),
                ToolResult(content=_text("# Page")),
            ]
        )
        _run(
            middleware,
            _context("goto", {"url": "https://example.org/a"}, server),
            ToolResult(content=_text("navigated")),
        )
        out = _run(
            middleware,
            _context("extract", server=server),
            ToolResult(content=_text("boom"), is_error=True),
        )

        assert server.calls == [
            ("markdown", {}, False),
            ("markdown", {"url": "https://example.org/a"}, False),
        ]
        assert out.content[1].text == "# Page"

    def test_failed_goto_is_not_remembered(self):
        middleware = ExtractMarkdownFallback()
        server = _FakeServer([ToolResult(content=_text("nope"), is_error=True)])
        _run(
            middleware,
            _context("goto", {"url": "https://example.org/gone"}, server),
            ToolResult(content=_text("navigation failed"), is_error=True),
        )
        out = _run(
            middleware,
            _context("extract", server=server),
            ToolResult(content=_text("boom"), is_error=True),
        )

        assert server.calls == [("markdown", {}, False)]
        assert out.is_error

    def test_original_error_is_returned_when_markdown_also_fails(self):
        server = _FakeServer([RuntimeError("markdown unavailable")])
        failure = ToolResult(content=_text("boom"), is_error=True)
        out = _run(ExtractMarkdownFallback(), _context("extract", server=server), failure)
        assert out is failure

    def test_empty_markdown_is_not_served_as_a_fallback(self):
        server = _FakeServer([ToolResult(content=[])])
        failure = ToolResult(content=_text("boom"), is_error=True)
        out = _run(ExtractMarkdownFallback(), _context("extract", server=server), failure)
        assert out is failure

    def test_blank_markdown_is_not_served_as_a_fallback(self):
        server = _FakeServer([ToolResult(content=_text("   "))])
        failure = ToolResult(content=_text("boom"), is_error=True)
        out = _run(ExtractMarkdownFallback(), _context("extract", server=server), failure)
        assert out is failure

    def test_erroring_markdown_is_not_served_as_a_fallback(self):
        server = _FakeServer([ToolResult(content=_text("no page"), is_error=True)])
        failure = ToolResult(content=_text("boom"), is_error=True)
        out = _run(ExtractMarkdownFallback(), _context("extract", server=server), failure)
        assert out is failure

    def test_missing_fastmcp_context_returns_the_original_error(self):
        failure = ToolResult(content=_text("boom"), is_error=True)
        context = MiddlewareContext(
            message=mt.CallToolRequestParams(name="extract", arguments={}),
            fastmcp_context=None,
        )
        assert _run(ExtractMarkdownFallback(), context, failure) is failure

    def test_goto_reporting_failure_in_its_text_is_not_remembered(self):
        middleware = ExtractMarkdownFallback()
        server = _FakeServer([ToolResult(content=_text("nope"), is_error=True)])
        _run(
            middleware,
            _context("goto", {"url": "https://example.org/tls"}, server),
            ToolResult(content=_text("SslConnectError on https://example.org/tls")),
        )
        out = _run(
            middleware,
            _context("extract", server=server),
            ToolResult(content=_text("boom"), is_error=True),
        )

        assert server.calls == [("markdown", {}, False)]
        assert out.is_error
