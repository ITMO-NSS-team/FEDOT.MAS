from typing import Literal

from fedotmas.mcp._config import HttpMCPServer, MCPServerConfig, StdioMCPServer
from fedotmas.mcp.discovery import discover_servers
from fedotmas.mcp.registry import create_toolset, get_mcp_servers, get_server_descriptions


def resolve_mcp_registry(
    mcp_servers: list[str] | dict[str, MCPServerConfig] | Literal["all"] | None,
) -> dict[str, MCPServerConfig]:
    """Resolve the user-facing *mcp_servers* argument into a registry dict."""
    if not mcp_servers:
        return {}
    if isinstance(mcp_servers, dict):
        return mcp_servers
    registry = get_mcp_servers()
    if mcp_servers == "all":
        return registry
    unknown = set(mcp_servers) - registry.keys()
    if unknown:
        raise ValueError(
            f"Unknown MCP servers: {sorted(unknown)}. "
            f"Available: {sorted(registry)}"
        )
    return {k: registry[k] for k in mcp_servers}


__all__ = [
    "HttpMCPServer",
    "MCPServerConfig",
    "StdioMCPServer",
    "create_toolset",
    "discover_servers",
    "get_mcp_servers",
    "get_server_descriptions",
    "resolve_mcp_registry",
]
