from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union


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
    """MCP server reachable over HTTP (Streamable HTTP transport)."""

    url: str
    headers: dict[str, str] = field(default_factory=dict)
    timeout: int = 60
    description: str = ""
    tags: tuple[str, ...] = ()


MCPServerConfig = Union[StdioMCPServer, HttpMCPServer]
