from __future__ import annotations

import os
from typing import Any

import mcp.types as mt
from fastmcp.client.transports import StdioTransport
from fastmcp.server import create_proxy
from fastmcp.server.middleware import CallNext, Middleware, MiddlewareContext
from fastmcp.tools import ToolResult

transport = StdioTransport(
    command="lightpanda",
    args=["mcp", "--insecure_disable_tls_host_verification"],
    env=dict(os.environ),
)

FALLBACK_NOTE = (
    "`extract` failed on this page: {error}\n"
    "The page itself follows as markdown. Read the values you need out of it; "
    "only if they are not there, navigate again or search."
)

# Lightpanda reports some navigation failures in the text of a result it does
# not mark as an error; fedotmas' BrowserFallbackPolicyPlugin matches the same
# strings, but this server cannot import from that package.
#: Soft cap on the markdown served as a rescue.  `extract` asks for fields, so
#: a caller that gets a whole page instead must at least get a bounded one: the
#: default plugin set carries no result truncation, and an unbounded render of
#: a long page either overruns the context window or spends the run's budget.
FALLBACK_MAX_BYTES = 30_000

#: Marks a result this middleware rescued.  It is still an error -- the schema
#: did not match -- but it carries the page, so a policy counting tool failures
#: should not hold it against the agent.  fedotmas' tool-error circuit breaker
#: reads the same key; the string is the contract between the two packages.
RESCUED_META_KEY = "fedotmas/rescued"

NAVIGATION_FAILURE_MARKERS = (
    "SslConnectError",
    "OperationTimedout",
    "UnsupportedProtocol",
    "navigate failed",
)


class ExtractMarkdownFallback(Middleware):
    """Append page markdown to a failed `extract`.

    `extract` raises inside lightpanda's own JavaScript whenever a selector in
    the caller's schema matches nothing (`Cannot read properties of null`) or
    the schema is malformed. The agent cannot tell those apart from a page that
    simply lacks the data, so it abandons the page and goes back to searching.
    Markdown of the page it already loaded answers the question it was asking.

    The result stays an error: flipping it to success would subject markdown's
    content to `extract`'s own output schema in the client
    (`mcp.client.session` validates only non-error results), and a caller that
    asked for fields should not silently receive a whole page as if the schema
    had matched.
    """

    def __init__(self) -> None:
        # Over stdio lightpanda refuses more than one session ("multiple
        # sessions require the HTTP transport"), so there is a single current
        # page and the last `goto` target is the page `extract` ran against.
        self._last_url: str | None = None

    async def on_call_tool(
        self,
        context: MiddlewareContext[mt.CallToolRequestParams],
        call_next: CallNext[mt.CallToolRequestParams, ToolResult],
    ) -> ToolResult:
        name = context.message.name
        arguments = context.message.arguments or {}
        result = await call_next(context)

        # Eleven of lightpanda's tools take a `url` and navigate before acting,
        # so tracking `goto` alone would point the retry at a stale page.
        url = arguments.get("url")
        if isinstance(url, str) and url and _navigation_succeeded(result):
            self._last_url = url

        if name != "extract" or not result.is_error:
            return result

        return await self._with_markdown(context, result)

    async def _with_markdown(
        self,
        context: MiddlewareContext[mt.CallToolRequestParams],
        failure: ToolResult,
    ) -> ToolResult:
        fastmcp_context = context.fastmcp_context
        if fastmcp_context is None:
            return failure
        server = fastmcp_context.fastmcp

        limits: dict[str, Any] = {
            "maxBytes": FALLBACK_MAX_BYTES,
            "strip": {"clutter": True},
        }
        attempts: list[dict[str, Any]] = [limits]
        if self._last_url:
            attempts.append({**limits, "url": self._last_url})

        for arguments in attempts:
            try:
                fallback = await server.call_tool(
                    "markdown", arguments, run_middleware=False
                )
            except Exception:
                continue
            if fallback.is_error or not _has_text(fallback):
                continue
            note = mt.TextContent(
                type="text", text=FALLBACK_NOTE.format(error=_error_text(failure))
            )
            meta = {**(failure.meta or {}), RESCUED_META_KEY: True}
            return ToolResult(
                content=[note, *fallback.content], meta=meta, is_error=True
            )

        return failure


def _text_blocks(result: ToolResult) -> list[str]:
    return [
        block.text.strip()
        for block in result.content
        if isinstance(block, mt.TextContent) and block.text.strip()
    ]


def _has_text(result: ToolResult) -> bool:
    return bool(_text_blocks(result))


def _error_text(result: ToolResult) -> str:
    return " ".join(_text_blocks(result)) or "no error message"


def _navigation_succeeded(result: ToolResult) -> bool:
    if result.is_error:
        return False
    text = " ".join(_text_blocks(result))
    return not any(marker in text for marker in NAVIGATION_FAILURE_MARKERS)


mcp = create_proxy(transport, name="web-scraping")
mcp.add_middleware(ExtractMarkdownFallback())


def main():
    mcp.run(show_banner=False)
