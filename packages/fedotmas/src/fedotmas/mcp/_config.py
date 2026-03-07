from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from typing import Union

_UV_BIN = shutil.which("uv") or "uv"  # try to find a path to `uv` binary


@dataclass(frozen=True)
class StdioMCPServer:
    """MCP server launched as a local subprocess (stdio transport)."""

    command: str
    args: tuple[str, ...]
    timeout: int = 60
    env: dict[str, str] = field(default_factory=dict)
    description: str = ""
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class HttpMCPServer:
    """MCP server reachable over HTTP/SSE."""

    url: str
    headers: dict[str, str] = field(default_factory=dict)
    timeout: int = 60
    description: str = ""
    tags: tuple[str, ...] = ()


MCPServerConfig = Union[StdioMCPServer, HttpMCPServer]


def directory_server(
    directory: str,
    entry_point: str,
    *,
    timeout: int = 60,
    description: str = "",
    tags: tuple[str, ...] = (),
) -> StdioMCPServer:
    """Local MCP server launched via ``uv run --directory``."""
    return StdioMCPServer(
        command=_UV_BIN,
        args=("run", "--directory", directory, entry_point),
        timeout=timeout,
        description=description,
        tags=tags,
    )


def workspace_server(
    package: str,
    entry_point: str,
    *,
    timeout: int = 60,
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
    timeout: int = 60,
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
    timeout: int = 60,
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
    timeout: int = 60,
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
