from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union

#: Seconds allowed a *locally spawned* MCP server, both to become ready and
#: for each subsequent tool call: ADK passes it to ``ClientSession`` as
#: ``read_timeout_seconds``.  Generous by default because a server whose venv
#: has not been built yet spends the first connection resolving dependencies --
#: run ``just mcp-sync`` to avoid paying that.  The cost of the generosity is
#: that a wedged tool call also stalls this long, which the pipeline-level
#: timeout is what ultimately bounds.  Does not apply to HTTP servers.
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
    #: ADK passes this to ``httpx.Timeout`` as the connect/write/pool budget
    #: for every request.  Kept tight where the stdio default is not, because
    #: an unreachable host should fail fast and no cold start is involved.
    timeout: int = 60
    description: str = ""
    tags: tuple[str, ...] = ()
    tool_name_prefix: str | None = None


MCPServerConfig = Union[StdioMCPServer, HttpMCPServer]
