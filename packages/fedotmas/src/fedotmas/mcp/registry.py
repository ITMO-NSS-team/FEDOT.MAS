from __future__ import annotations

import asyncio
import functools
import os

from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.mcp_tool import (
    McpToolset,
    StdioConnectionParams,
    StreamableHTTPConnectionParams,
)
from mcp import StdioServerParameters
from mcp.client.stdio import get_default_environment

from fedotmas.common.logging import get_logger
from fedotmas.mcp._config import (
    DEFAULT_MCP_TIMEOUT_S,
    HttpMCPServer,
    MCPServerConfig,
    StdioMCPServer,
)
from fedotmas.mcp.capabilities import is_solving_tool
from fedotmas.mcp.discovery import discover_local_servers

_log = get_logger("fedotmas.mcp.registry")

#: Variables pointing at *our* virtualenv.  Local servers run under ``uv run
#: --directory``, which resolves its own; an inherited value makes uv warn and
#: can send the child at the wrong .venv.
_PARENT_VENV_VARS = frozenset({"VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT"})


@functools.cache
def get_mcp_servers() -> dict[str, MCPServerConfig]:
    """Cached registry of MCP servers discovered from ``<workspace_root>/mcp-servers``."""
    return discover_local_servers()


def create_toolset(
    name: str,
    registry: dict[str, MCPServerConfig] | None = None,
    *,
    include_diagnostic: bool = False,
) -> McpToolset:
    """Create a toolset, hiding diagnostic tools from solving agents by default."""
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

    filtered_diagnostics: set[str] = set()

    def solving_tool_filter(
        tool: BaseTool, readonly_context: ReadonlyContext | None = None
    ) -> bool:
        tool_name = getattr(tool, "name", "")
        visible = is_solving_tool(f"{name}__{tool_name}")
        if not visible:
            filtered_diagnostics.add(tool_name)
        return visible

    toolset = McpToolset(
        connection_params=params,
        tool_filter=None if include_diagnostic else solving_tool_filter,
        tool_name_prefix=cfg.tool_name_prefix,
    )
    # Used only for bounded turn-observability records; the predicate itself
    # owns filtering and the MCP server continues to expose its internal tool.
    toolset._fedotmas_filtered_diagnostic_tools = filtered_diagnostics
    return toolset


async def list_server_tools(
    name: str,
    registry: dict[str, MCPServerConfig] | None = None,
    *,
    timeout: float | None = None,
    include_diagnostic: bool = False,
) -> list[BaseTool]:
    """Connect to the named server and return the tools it advertises.

    Raises whatever the connection raises, ``TimeoutError`` included; callers
    decide what an unreachable server means.  *timeout* defaults to the
    server's own, which is already tight for a URL server and generous for a
    local one; a caller that has to tolerate a cold venv build adds its own
    grace on top.
    """
    if timeout is None:
        reg = registry if registry is not None else get_mcp_servers()
        timeout = getattr(reg.get(name), "timeout", DEFAULT_MCP_TIMEOUT_S)

    toolset = create_toolset(
        name, registry=registry, include_diagnostic=include_diagnostic
    )
    try:
        return list(await asyncio.wait_for(toolset.get_tools(), timeout))
    finally:
        try:
            await toolset.close()
        except Exception as exc:  # noqa: BLE001 - closing must not mask tool listing
            _log.debug("Closing toolset for '{}' failed: {}", name, exc)


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

    Only prefixes from the auto-discovered registry are known here; one declared
    in a registry passed straight to ``MAS``/``MAW`` is not, and such a tool
    keeps its prefixed name.
    """
    try:
        registry = get_mcp_servers()
    except Exception as exc:  # noqa: BLE001 - installed use may lack a workspace root
        # Discovery raises without a workspace root -- the installed-as-a-
        # dependency shape.  On the per-tool-call path, so it must not raise.
        _log.debug(
            "Cannot resolve tool name prefixes ({}); using '{}' as is", exc, tool_name
        )
        return tool_name

    for cfg in registry.values():
        prefix = cfg.tool_name_prefix
        if prefix and tool_name.startswith(f"{prefix}_"):
            return tool_name[len(prefix) + 1 :]
    return tool_name
