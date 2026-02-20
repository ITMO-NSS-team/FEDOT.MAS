from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from typing import Union

from google.adk.tools.mcp_tool import (
    McpToolset,
    SseConnectionParams,
    StdioConnectionParams,
)
from mcp import StdioServerParameters

from fedotmas.common.logging import get_logger

_log = get_logger("fedotmas.mcp.registry")

_UV_BIN = shutil.which("uv") or "uv"


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StdioMCPServer:
    """MCP server launched as a local subprocess (stdio transport)."""

    command: str
    args: tuple[str, ...]
    timeout: int = 10
    env: dict[str, str] = field(default_factory=dict)
    description: str = ""
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class HttpMCPServer:
    """MCP server reachable over HTTP/SSE."""

    url: str
    headers: dict[str, str] = field(default_factory=dict)
    timeout: int = 10
    description: str = ""
    tags: tuple[str, ...] = ()


MCPServerConfig = Union[StdioMCPServer, HttpMCPServer]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def workspace_server(
    package: str,
    entry_point: str,
    *,
    timeout: int = 10,
    description: str = "",
    tags: tuple[str, ...] = (),
) -> StdioMCPServer:
    """Config for a workspace-local MCP server package launched via ``uv run``."""
    return StdioMCPServer(
        command=_UV_BIN,
        args=("run", "--package", package, entry_point),
        timeout=timeout,
        description=description,
        tags=tags,
    )


def npx_server(
    package: str,
    *,
    timeout: int = 10,
    extra_args: list[str] | None = None,
    description: str = "",
    tags: tuple[str, ...] = (),
) -> StdioMCPServer:
    base_args = ["-y", package]
    if extra_args:
        base_args.extend(extra_args)
    return StdioMCPServer(
        command="npx",
        args=tuple(base_args),
        timeout=timeout,
        description=description,
        tags=tags,
    )


def uvx_server(
    package: str,
    *,
    timeout: int = 10,
    extra_args: list[str] | None = None,
    description: str = "",
    tags: tuple[str, ...] = (),
) -> StdioMCPServer:
    args = tuple(extra_args) if extra_args else ("--from", package, package)
    return StdioMCPServer(
        command="uvx",
        args=args,
        timeout=timeout,
        description=description,
        tags=tags,
    )


def http_server(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: int = 10,
    description: str = "",
    tags: tuple[str, ...] = (),
) -> HttpMCPServer:
    """Config for a remote MCP server reachable over HTTP/SSE."""
    return HttpMCPServer(
        url=url,
        headers=headers or {},
        timeout=timeout,
        description=description,
        tags=tags,
    )


# ---------------------------------------------------------------------------
# Default registry
# ---------------------------------------------------------------------------

MCP_SERVERS: dict[str, MCPServerConfig] = {
    "download-url-content": workspace_server(
        package="fedotmas-mcp-download",
        entry_point="fedotmas-mcp-download",
        description=(
            "Download files from URLs to the local filesystem. "
            "Supports single and batch downloads with size validation."
        ),
        tags=("web", "files"),
    ),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


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
    reg = registry or MCP_SERVERS
    if tags:
        reg = {k: v for k, v in reg.items() if tags & set(v.tags)}
    return {name: cfg.description or f"MCP server: {name}" for name, cfg in reg.items()}
