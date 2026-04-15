from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from fedotmas.mcp._config import MCPServerConfig


@dataclass
class ToolDescriptor:
    """Framework-neutral description of a tool available to an agent.

    Exactly one of *mcp_server* or *function* should be set.

    For MCP tools, *mcp_server_name* and *mcp_server* describe the server
    to connect to.  For function tools, *function* is the callable.
    """

    name: str
    mcp_server_name: str | None = None
    mcp_server: MCPServerConfig | None = None
    function: Callable[..., Any] | None = None
    description: str = ""
    after_tool_callback: Callable[..., Any] | None = None
