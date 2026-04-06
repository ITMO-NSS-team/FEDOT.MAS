from __future__ import annotations

import os

from fastmcp.client.transports import StdioTransport
from fastmcp.server import create_proxy

transport = StdioTransport(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-sequential-thinking"],
    env=dict(os.environ),
)

mcp = create_proxy(transport, name="sequential-thinking")


def main():
    mcp.run(show_banner=False)
