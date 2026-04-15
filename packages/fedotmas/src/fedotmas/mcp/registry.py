from __future__ import annotations

import functools

from fedotmas.common.logging import get_logger
from fedotmas.interfaces.tools import ToolDescriptor
from fedotmas.mcp._config import MCPServerConfig
from fedotmas.mcp.discovery import discover_local_servers

_log = get_logger("fedotmas.mcp.registry")


@functools.cache
def get_mcp_servers() -> dict[str, MCPServerConfig]:
    """Cached registry of MCP servers discovered from ``<workspace_root>/mcp-servers``."""
    return discover_local_servers()


def create_toolset(
    name: str, registry: dict[str, MCPServerConfig] | None = None
) -> ToolDescriptor:
    """Create a framework-neutral ``ToolDescriptor`` for the named MCP server."""
    reg = registry if registry is not None else get_mcp_servers()
    if name not in reg:
        _log.error("Unknown MCP server: '{}' | available={}", name, sorted(reg))
        raise ValueError(f"Unknown MCP server: '{name}'. Available: {sorted(reg)}")

    cfg = reg[name]
    _log.debug("Creating MCP tool descriptor | server={}", name)
    return ToolDescriptor(
        name=name,
        mcp_server_name=name,
        mcp_server=cfg,
        description=cfg.description,
    )


def get_server_descriptions(
    registry: dict[str, MCPServerConfig] | None = None,
    *,
    tags: set[str] | None = None,
) -> dict[str, str]:
    """Return ``{name: description}`` for every registered server.

    If *tags* is given, only servers whose tags overlap with the
    requested set are included.
    """
    reg = registry if registry is not None else get_mcp_servers()
    if tags:
        reg = {k: v for k, v in reg.items() if tags & set(v.tags)}
    return {name: cfg.description or f"MCP server: {name}" for name, cfg in reg.items()}
