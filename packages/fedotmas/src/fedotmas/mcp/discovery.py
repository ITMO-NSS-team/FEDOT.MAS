from __future__ import annotations

import os
import re
import tomllib
from pathlib import Path

from fedotmas.common.logging import get_logger
from fedotmas.mcp._config import (
    DEFAULT_MCP_TIMEOUT_S,
    MCPServerConfig,
    StdioMCPServer,
)

_log = get_logger("fedotmas.mcp.discovery")

_UV_BIN: str | None = None

#: Intersection of what the providers accept in a function name.  Gemini
#: also allows ".", OpenAI-compatible endpoints do not, and this repo
#: defaults to the latter -- so take the stricter set.
_PREFIX_CHARS = re.compile(r"[A-Za-z0-9_-]+")

#: Providers cap a function name at 64 characters, and the prefix has to
#: leave room for the tool name it is prepended to.
_MAX_PREFIX_LEN = 24


def _get_uv_bin() -> str:
    global _UV_BIN
    if _UV_BIN is None:
        import shutil

        _UV_BIN = shutil.which("uv") or "uv"
    return _UV_BIN


def _default_timeout() -> int:
    """Session-ready timeout for servers that do not declare one."""
    value = os.getenv("FEDOTMAS_MCP_TIMEOUT_S")
    if value is None:
        return DEFAULT_MCP_TIMEOUT_S
    try:
        parsed = int(value)
    except ValueError:
        parsed = 0
    if parsed <= 0:
        _log.warning(
            "Invalid FEDOTMAS_MCP_TIMEOUT_S={!r}; using {}",
            value,
            DEFAULT_MCP_TIMEOUT_S,
        )
        return DEFAULT_MCP_TIMEOUT_S
    return parsed


def _resolve_prefix(value: object, server_name: str) -> str | None:
    """Validate a declared ``tool_name_prefix``.

    ADK joins it as ``f"{prefix}_{tool.name}"``, and a function name has to
    survive every provider it is sent to.  Only letters, digits, ``-`` and
    ``_`` are accepted -- narrower than any single provider allows, since a
    prefix rejected here costs a line in a pyproject, while one wrongly
    accepted surfaces as an opaque 400 at the first model call.
    """
    if value is None or value == "":
        return None
    if not isinstance(value, str) or not _PREFIX_CHARS.fullmatch(value):
        _log.warning(
            "Ignoring invalid tool_name_prefix={!r} for server '{}'; "
            "expected a string of letters, digits, '-' or '_'",
            value,
            server_name,
        )
        return None
    if len(value) > _MAX_PREFIX_LEN:
        _log.warning(
            "Ignoring tool_name_prefix={!r} for server '{}'; longer than {} "
            "characters leaves too little of the 64-character name budget",
            value,
            server_name,
            _MAX_PREFIX_LEN,
        )
        return None
    return value


def _directory_server(
    directory: str,
    entry_point: str,
    *,
    timeout: int = DEFAULT_MCP_TIMEOUT_S,
    description: str = "",
    tags: tuple[str, ...] = (),
    tool_name_prefix: str | None = None,
) -> StdioMCPServer:
    """Local MCP server launched via ``uv run --directory``."""
    return StdioMCPServer(
        command=_get_uv_bin(),
        args=("run", "--directory", directory, entry_point),
        timeout=timeout,
        description=description,
        tags=tags,
        tool_name_prefix=tool_name_prefix,
    )


def _find_repo_root() -> Path:
    """Walk up from this file to find the workspace root (contains pyproject.toml with [tool.uv.workspace])."""
    current = Path(__file__).resolve().parent
    for parent in (current, *current.parents):
        candidate = parent / "pyproject.toml"
        if candidate.is_file():
            data = tomllib.loads(candidate.read_text())
            if (
                "tool" in data
                and "uv" in data["tool"]
                and "workspace" in data["tool"]["uv"]
            ):
                return parent
    msg = "Could not find workspace root (pyproject.toml with [tool.uv.workspace])"
    raise RuntimeError(msg)


def discover_local_servers(
    servers_dir: str | Path | None = None,
) -> dict[str, MCPServerConfig]:
    """Scan a directory of MCP server packages and return a registry.

    Args:
        servers_dir: Absolute path to the directory containing server
            sub-directories.  Each sub-directory must have a ``pyproject.toml``.
            Defaults to ``<workspace_root>/mcp-servers``.

    Expected layout::

        servers_dir/
        ├── my-server/
        │   ├── pyproject.toml   # must contain [tool.fedotmas.mcp] and [project.scripts]
        │   └── ...
        └── another-server/
            ├── pyproject.toml
            └── ...

    Each ``pyproject.toml`` must declare ``[tool.fedotmas.mcp]`` with at least
    ``name`` (str).  Optional: ``description``, ``tags``, ``timeout``,
    ``tool_name_prefix``.

    A server that does not declare ``timeout`` gets ``FEDOTMAS_MCP_TIMEOUT_S``
    if set, else :data:`DEFAULT_MCP_TIMEOUT_S`.

    Server resolution order:

    1. If ``mcp.command`` is present → use it directly as an external binary
       (e.g. ``command = "lightpanda"``, ``args = ["mcp"]``).
    2. Otherwise fall back to ``[project.scripts]`` + ``uv run --directory``.

    Returns:
        ``{name: MCPServerConfig}`` for every successfully discovered server.
    """
    if servers_dir is not None:
        servers_dir = Path(servers_dir)
    else:
        repo_root = _find_repo_root()
        servers_dir = repo_root / "mcp-servers"

    if not servers_dir.is_dir():
        _log.warning("MCP servers directory not found: {}", servers_dir)
        return {}

    result: dict[str, MCPServerConfig] = {}
    default_timeout = _default_timeout()

    for pyproject_path in sorted(servers_dir.glob("*/pyproject.toml")):
        server_dir = pyproject_path.parent
        try:
            data = tomllib.loads(pyproject_path.read_text())
        except (tomllib.TOMLDecodeError, OSError) as e:
            _log.warning("Failed to parse {}: {}", pyproject_path, e)
            continue

        mcp_meta = data.get("tool", {}).get("fedotmas", {}).get("mcp")
        if mcp_meta is None:
            continue

        name = mcp_meta.get("name")
        if not name:
            _log.warning("Missing 'name' in [tool.fedotmas.mcp] of {}", pyproject_path)
            continue

        description = mcp_meta.get("description", "")
        tags = tuple(mcp_meta.get("tags", ()))
        timeout = mcp_meta.get("timeout")
        tool_name_prefix = _resolve_prefix(mcp_meta.get("tool_name_prefix"), name)

        command = mcp_meta.get("command")
        if command:
            result[name] = StdioMCPServer(
                command=str(command),
                args=tuple(mcp_meta.get("args", ())),
                timeout=int(timeout) if timeout is not None else default_timeout,
                description=str(description),
                tags=tags,
                tool_name_prefix=tool_name_prefix,
            )
            _log.debug("Discovered external MCP server: {} -> {}", name, command)
            continue

        scripts = data.get("project", {}).get("scripts", {})
        if not scripts:
            _log.warning("No [project.scripts] in {}", pyproject_path)
            continue
        entry_point = next(iter(scripts))

        result[name] = _directory_server(
            directory=str(server_dir),
            entry_point=str(entry_point),
            timeout=int(timeout) if timeout is not None else default_timeout,
            description=str(description),
            tags=tags,
            tool_name_prefix=tool_name_prefix,
        )
        _log.debug("Discovered MCP server: {} -> {}", name, server_dir)

    return result
