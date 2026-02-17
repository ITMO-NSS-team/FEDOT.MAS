from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

from google.adk.tools.mcp_tool import McpToolset, StdioConnectionParams
from mcp import StdioServerParameters

from fedotmas.common.logging import get_logger

_log = get_logger("fedotmas.mcp.registry")

_SERVERS_DIR = Path(__file__).resolve().parent / "servers"


@dataclass(frozen=True)
class MCPServerConfig:
    """Declarative config for an MCP server."""

    command: str
    args: tuple[str, ...]
    timeout: int = 10
    env: dict[str, str] = field(default_factory=dict)
    description: str = ""


# Helpers


def python_server(
    relative_path: str,
    *,
    timeout: int = 10,
    description: str = "",
) -> MCPServerConfig:
    """Config for a local Python MCP server under ``mcp/servers/``."""
    return MCPServerConfig(
        command=sys.executable,
        args=(str(_SERVERS_DIR / relative_path),),
        timeout=timeout,
        description=description,
    )


def npx_server(
    package: str,
    *,
    timeout: int = 10,
    extra_args: list[str] | None = None,
    description: str = "",
) -> MCPServerConfig:
    base_args = ["-y", package]
    if extra_args:
        base_args.extend(extra_args)
    return MCPServerConfig(
        command="npx",
        args=tuple(base_args),
        timeout=timeout,
        description=description,
    )


def uvx_server(
    package: str,
    *,
    timeout: int = 10,
    extra_args: list[str] | None = None,
    description: str = "",
) -> MCPServerConfig:
    args = tuple(extra_args) if extra_args else ("--from", package, package)
    return MCPServerConfig(
        command="uvx",
        args=args,
        timeout=timeout,
        description=description,
    )


# ---------------------------------------------------------------------------
# Default registry
# ---------------------------------------------------------------------------

MCP_SERVERS: dict[str, MCPServerConfig] = {
    "download-url-content": python_server(
        "download/server.py",
        timeout=30,
        description=(
            "Download files from URLs to the local filesystem. "
            "Supports single and batch downloads with size validation."
        ),
    ),
}


# Public API


def create_toolset(
    name: str, registry: dict[str, MCPServerConfig] | None = None
) -> McpToolset:
    """Create an ADK ``McpToolset`` for the named server."""
    reg = registry or MCP_SERVERS
    if name not in reg:
        _log.error("Unknown MCP server: '{}' | available={}", name, sorted(reg))
        raise ValueError(f"Unknown MCP server: '{name}'. Available: {sorted(reg)}")

    cfg = reg[name]
    _log.debug("Creating MCP toolset | server={}", name)
    return McpToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(
                command=cfg.command,
                args=list(cfg.args),
                env=cfg.env or None,
            ),
        ),
    )


def get_server_descriptions(
    registry: dict[str, MCPServerConfig] | None = None,
) -> dict[str, str]:
    """Return ``{name: description}`` for every registered server."""
    reg = registry or MCP_SERVERS
    return {name: cfg.description or f"MCP server: {name}" for name, cfg in reg.items()}
