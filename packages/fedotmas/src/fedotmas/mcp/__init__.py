from typing import Literal

from fedotmas.mcp._config import HttpMCPServer, MCPServerConfig, StdioMCPServer
from fedotmas.mcp._discovery import discover_servers
from fedotmas.mcp.registry import MCP_SERVERS, create_toolset, get_server_descriptions

ALL_SERVERS: list[str] = list(MCP_SERVERS.keys())


def resolve_mcp_registry(
    mcp_servers: list[str] | dict[str, MCPServerConfig] | Literal["all"],
) -> dict[str, MCPServerConfig]:
    """Resolve the user-facing *mcp_servers* argument into a registry dict."""
    if mcp_servers == "all":
        return MCP_SERVERS
    if isinstance(mcp_servers, dict):
        return mcp_servers
    # list[str] — filter from the bank (empty list = no tools)
    unknown = set(mcp_servers) - MCP_SERVERS.keys()
    if unknown:
        raise ValueError(
            f"Unknown MCP servers: {sorted(unknown)}. "
            f"Available: {sorted(MCP_SERVERS)}"
        )
    return {k: MCP_SERVERS[k] for k in mcp_servers}


__all__ = [
    "ALL_SERVERS",
    "HttpMCPServer",
    "MCPServerConfig",
    "MCP_SERVERS",
    "StdioMCPServer",
    "create_toolset",
    "discover_servers",
    "get_server_descriptions",
    "resolve_mcp_registry",
]
