from __future__ import annotations

import functools

from google.adk.tools.mcp_tool import (
    McpToolset,
    SseConnectionParams,
    StdioConnectionParams,
)
from mcp import StdioServerParameters

from fedotmas.common.logging import get_logger
from fedotmas.mcp._config import HttpMCPServer, MCPServerConfig, StdioMCPServer
from fedotmas.mcp._discovery import discover_servers

_log = get_logger("fedotmas.mcp.registry")


@functools.cache
def get_mcp_servers() -> dict[str, MCPServerConfig]:
    """Discover and return the MCP server registry."""
    return discover_servers()


def create_toolset(
    name: str, registry: dict[str, MCPServerConfig] | None = None
) -> McpToolset:
    """Create an ADK ``McpToolset`` for the named server."""
    reg = registry or get_mcp_servers()
    if name not in reg:
        _log.error("Unknown MCP server: '{}' | available={}", name, sorted(reg))
        raise ValueError(f"Unknown MCP server: '{name}'. Available: {sorted(reg)}")

    cfg = reg[name]
    _log.debug("Creating MCP toolset | server={}", name)

    match cfg:
        case StdioMCPServer():
            params = StdioConnectionParams(
                server_params=StdioServerParameters(
                    command=cfg.command,
                    args=list(cfg.args),
                    env=cfg.env or None,
                ),
                timeout=cfg.timeout,
            )
        case HttpMCPServer():
            params = SseConnectionParams(
                url=cfg.url,
                headers=cfg.headers or None,
                timeout=cfg.timeout,
            )
        case _:
            raise TypeError(f"Unsupported MCP server type: {type(cfg)}")

    return McpToolset(connection_params=params)


def get_server_descriptions(
    registry: dict[str, MCPServerConfig] | None = None,
    *,
    tags: set[str] | None = None,
) -> dict[str, str]:
    """Return ``{name: description}`` for every registered server.

    If *tags* is given, only servers whose tags overlap with the
    requested set are included.
    """
    reg = registry or get_mcp_servers()
    if tags:
        reg = {k: v for k, v in reg.items() if tags & set(v.tags)}
    return {name: cfg.description or f"MCP server: {name}" for name, cfg in reg.items()}
