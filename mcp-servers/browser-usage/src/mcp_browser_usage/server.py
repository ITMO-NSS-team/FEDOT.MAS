from __future__ import annotations

import os

from fastmcp.client.transports import StdioTransport
from fastmcp.server import create_proxy

_env = dict(os.environ)
_env.setdefault("BROWSER_USE_HEADLESS", "true")
_env.setdefault("BROWSER_USE_LLM_MODEL", "openai/gpt-4o-mini")

transport = StdioTransport(
    command="uvx",
    args=["--from", "browser-use[cli]", "browser-use", "--mcp"],
    env=_env,
)

mcp = create_proxy(transport, name="browser-usage")


def main():
    mcp.run(show_banner=False)
