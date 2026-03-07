from __future__ import annotations

import tomllib
from pathlib import Path

from fedotmas.common.logging import get_logger
from fedotmas.mcp._config import MCPServerConfig, StdioMCPServer

_log = get_logger("fedotmas.mcp.discovery")

_UV_BIN: str | None = None


def _get_uv_bin() -> str:
    global _UV_BIN
    if _UV_BIN is None:
        import shutil

        _UV_BIN = shutil.which("uv") or "uv"
    return _UV_BIN


def _directory_server(
    directory: str,
    entry_point: str,
    *,
    timeout: int = 60,
    description: str = "",
    tags: tuple[str, ...] = (),
) -> StdioMCPServer:
    """Local MCP server launched via ``uv run --directory``."""
    return StdioMCPServer(
        command=_get_uv_bin(),
        args=("run", "--directory", directory, entry_point),
        timeout=timeout,
        description=description,
        tags=tags,
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

    Each ``pyproject.toml`` must declare:

    - ``[tool.fedotmas.mcp]`` with at least ``name`` (str).
      Optional: ``description`` (str), ``tags`` (list[str]).
    - ``[project.scripts]`` — the first entry is used as the server entry point.

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

        # Get entry point from [project.scripts]
        scripts = data.get("project", {}).get("scripts", {})
        if not scripts:
            _log.warning("No [project.scripts] in {}", pyproject_path)
            continue
        entry_point = next(iter(scripts))

        description = mcp_meta.get("description", "")
        tags = tuple(mcp_meta.get("tags", ()))
        timeout = mcp_meta.get("timeout")

        kwargs: dict[str, object] = dict(
            directory=str(server_dir),
            entry_point=entry_point,
            description=description,
            tags=tags,
        )
        if timeout is not None:
            kwargs["timeout"] = int(timeout)

        result[name] = _directory_server(**kwargs)  # type: ignore[arg-type]
        _log.debug("Discovered MCP server: {} -> {}", name, server_dir)

    return result
