from __future__ import annotations

import base64
import binascii
import json
import os
import sys
import tempfile
import uuid
from pathlib import Path

import mcp.types as mt
from fastmcp.client.transports import StdioTransport
from fastmcp.server import create_proxy
from fastmcp.server.middleware import CallNext, Middleware, MiddlewareContext
from fastmcp.tools import ToolResult

_env = dict(os.environ)
_env.setdefault("BROWSER_USE_HEADLESS", "true")
_env.setdefault("BROWSER_USE_LLM_MODEL", "openai/gpt-4o-mini")

UPSTREAM = ["uvx", "--from", "browser-use[cli]", "browser-use", "--mcp"]

# Through the guard: otherwise browser-use and its Chrome outlive the proxy
# (see _guard's docstring).
transport = StdioTransport(
    command=sys.executable,
    args=["-m", "mcp_browser_usage._guard", str(os.getpid()), *UPSTREAM],
    env=_env,
)

SCREENSHOT_TOOLS = frozenset({"browser_screenshot", "browser_get_state"})

SCREENSHOT_NOTE = (
    "The screenshot is saved at screenshot_path; pass that path to the media "
    "server's analyze_image to look at it."
)


class ScreenshotToFile(Middleware):
    """Replace base64 screenshots in browser-use results with a file path.

    browser-use returns the screenshot as a base64 field inside a JSON text
    result, not as an MCP image, so the calling model receives hundreds of
    kilobytes of base64 as plain text and cannot see the picture anyway;
    gpt-5-mini answered such a turn with no content at all.  A file path is
    something the media server's analyze_image can open.
    """

    def __init__(self, directory: Path) -> None:
        self._directory = directory

    async def on_call_tool(
        self,
        context: MiddlewareContext[mt.CallToolRequestParams],
        call_next: CallNext[mt.CallToolRequestParams, ToolResult],
    ) -> ToolResult:
        result = await call_next(context)
        if context.message.name not in SCREENSHOT_TOOLS or result.is_error:
            return result

        content = [self._rewrite(block) for block in result.content]
        return ToolResult(
            content=content,
            structured_content=result.structured_content,
            meta=result.meta,
        )

    def _rewrite(self, block: mt.ContentBlock) -> mt.ContentBlock:
        if not isinstance(block, mt.TextContent):
            return block
        try:
            payload = json.loads(block.text)
        except ValueError:
            return block
        if not isinstance(payload, dict) or not isinstance(
            payload.get("screenshot"), str
        ):
            return block

        encoded = payload.pop("screenshot")
        try:
            data = base64.b64decode(encoded, validate=True)
        except binascii.Error:
            payload["screenshot_error"] = "browser returned an undecodable screenshot"
        else:
            payload["screenshot_path"] = str(self._save(data))
            payload["screenshot_note"] = SCREENSHOT_NOTE
        return mt.TextContent(type="text", text=json.dumps(payload, indent=2))

    def _save(self, data: bytes) -> Path:
        suffix = ".jpg" if data.startswith(b"\xff\xd8") else ".png"
        self._directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        path = self._directory / f"screenshot-{uuid.uuid4().hex}{suffix}"
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as file:
            file.write(data)
        return path


def _screenshot_dir() -> Path:
    configured = os.environ.get("BROWSER_USAGE_SCREENSHOT_DIR")
    if configured:
        return Path(configured)
    return Path(tempfile.gettempdir()) / "fedotmas-screenshots"


mcp = create_proxy(transport, name="browser-usage")
mcp.add_middleware(ScreenshotToFile(_screenshot_dir()))


def main():
    mcp.run(show_banner=False)
