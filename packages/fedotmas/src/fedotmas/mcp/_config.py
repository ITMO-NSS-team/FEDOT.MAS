from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union

#: Seconds allowed a *locally spawned* MCP server, both to become ready and for
#: each subsequent tool call (ADK passes it as ``read_timeout_seconds``).
#: Generous because a server whose venv is unbuilt resolves dependencies on the
#: first connection -- ``just mcp-sync`` avoids paying that.  The cost is that a
#: wedged call stalls this long, bounded only by the pipeline timeout.
DEFAULT_MCP_TIMEOUT_S = 180


@dataclass(frozen=True)
class StdioMCPServer:
    """MCP server launched as a local subprocess (stdio transport)."""

    command: str
    args: tuple[str, ...]
    timeout: int = DEFAULT_MCP_TIMEOUT_S
    env: dict[str, str] = field(default_factory=dict)
    description: str = ""
    tags: tuple[str, ...] = ()
    #: Set only on a server whose tool names collide with another's; ADK joins
    #: it as ``f"{prefix}_{tool.name}"``.  Unset elsewhere on purpose: models
    #: call plain names like ``search`` more reliably than decorated ones.
    tool_name_prefix: str | None = None


@dataclass(frozen=True)
class HttpMCPServer:
    """MCP server reachable over HTTP (Streamable HTTP transport)."""

    url: str
    headers: dict[str, str] = field(default_factory=dict)
    #: ADK passes this to ``httpx.Timeout`` per request.  Tight where the stdio
    #: default is not: no cold start is involved, so an unreachable host should
    #: fail fast.
    timeout: int = 60
    description: str = ""
    tags: tuple[str, ...] = ()
    tool_name_prefix: str | None = None


MCPServerConfig = Union[StdioMCPServer, HttpMCPServer]
