from __future__ import annotations

import functools
import os

from google.adk.tools.mcp_tool import (
    McpToolset,
    StdioConnectionParams,
    StreamableHTTPConnectionParams,
)
from mcp import StdioServerParameters
from mcp.client.stdio import get_default_environment

from fedotmas.common.logging import get_logger
from fedotmas.mcp._config import HttpMCPServer, MCPServerConfig, StdioMCPServer
from fedotmas.mcp.discovery import discover_local_servers

_log = get_logger("fedotmas.mcp.registry")

#: Variables that point at *our* virtualenv.  Local servers are launched with
#: ``uv run --directory``, which resolves its own environment per server; an
#: inherited value makes uv warn and can send the child at the wrong .venv.
_PARENT_VENV_VARS = frozenset({"VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT"})


@functools.cache
def get_mcp_servers() -> dict[str, MCPServerConfig]:
    """Cached registry of MCP servers discovered from ``<workspace_root>/mcp-servers``."""
    return discover_local_servers()


def create_toolset(
    name: str, registry: dict[str, MCPServerConfig] | None = None
) -> McpToolset:
    """Create an ADK ``McpToolset`` for the named server."""
    reg = registry if registry is not None else get_mcp_servers()
    if name not in reg:
        _log.error("Unknown MCP server: '{}' | available={}", name, sorted(reg))
        raise ValueError(f"Unknown MCP server: '{name}'. Available: {sorted(reg)}")

    cfg = reg[name]
    _log.debug("Creating MCP toolset | server={}", name)

    match cfg:
        case StdioMCPServer():
            inherited = {
                k: v for k, v in os.environ.items() if k not in _PARENT_VENV_VARS
            }
            env = {**get_default_environment(), **inherited, **cfg.env}
            params = StdioConnectionParams(
                server_params=StdioServerParameters(
                    command=cfg.command,
                    args=list(cfg.args),
                    env=env,
                ),
                timeout=cfg.timeout,
            )
        case HttpMCPServer():
            params = StreamableHTTPConnectionParams(
                url=cfg.url,
                headers=cfg.headers or None,
                timeout=cfg.timeout,
            )
        case _:
            raise TypeError(f"Unsupported MCP server type: {type(cfg)}")

    return McpToolset(
        connection_params=params,
        tool_name_prefix=cfg.tool_name_prefix,
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


def strip_tool_name_prefix(tool_name: str) -> str:
    """Return *tool_name* without a registered server's ``tool_name_prefix``.

    A prefixed server renames every one of its tools, which silently breaks
    any policy that matches tool names literally.  Callers that reason about
    tool identity should compare against this rather than the raw name.
    """
    for cfg in get_mcp_servers().values():
        prefix = cfg.tool_name_prefix
        if prefix and tool_name.startswith(f"{prefix}_"):
            return tool_name[len(prefix) + 1 :]
    return tool_name
