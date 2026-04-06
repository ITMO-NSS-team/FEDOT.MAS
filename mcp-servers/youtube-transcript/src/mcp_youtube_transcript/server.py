from __future__ import annotations

import os

from fastmcp.client.transports import StdioTransport
from fastmcp.server import create_proxy

transport = StdioTransport(
    command="uvx",
    args=[
        "--from",
        "git+https://github.com/jkawamoto/mcp-youtube-transcript",
        "mcp-youtube-transcript",
    ],
    env=dict(os.environ),
)

mcp = create_proxy(transport, name="youtube-transcript")


def main():
    mcp.run(show_banner=False)
