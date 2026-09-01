from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union

#: Seconds to wait for a *locally spawned* MCP session to become ready.
#: Generous by default: a server whose venv has not been built yet spends the
#: first connection resolving dependencies.  Run ``just mcp-sync`` to avoid
#: paying that here.  Does not apply to HTTP servers -- see below.
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
    #: Set only on a server whose tool names collide with another's.  ADK joins
    #: it as ``f"{prefix}_{tool.name}"``.  Left unset everywhere else on purpose:
    #: models call plain names like ``search`` far more reliably than decorated
    #: ones, so renaming is a targeted fix, not a blanket policy.
    tool_name_prefix: str | None = None


@dataclass(frozen=True)
class HttpMCPServer:
    """MCP server reachable over HTTP (Streamable HTTP transport)."""

    url: str
    headers: dict[str, str] = field(default_factory=dict)
    #: ADK passes this to ``httpx.Timeout`` as the connect/write/pool budget for
    #: *every* request, not just session setup, so it stays tight: an
    #: unreachable host should fail fast rather than stall the agent.
    timeout: int = 60
    description: str = ""
    tags: tuple[str, ...] = ()
    tool_name_prefix: str | None = None


MCPServerConfig = Union[StdioMCPServer, HttpMCPServer]
