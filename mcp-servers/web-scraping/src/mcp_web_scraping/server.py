from __future__ import annotations

import os

from fastmcp.client.transports import StdioTransport
from fastmcp.server import create_proxy

transport = StdioTransport(
    command="lightpanda",
    args=["mcp", "--insecure_disable_tls_host_verification"],
    env=dict(os.environ),
)

mcp = create_proxy(transport, name="web-scraping")


def main():
    mcp.run(show_banner=False)
