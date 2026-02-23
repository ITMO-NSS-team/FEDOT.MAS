from __future__ import annotations

import tomllib
from pathlib import Path

from fedotmas.common.logging import get_logger
from fedotmas.mcp._config import MCPServerConfig, directory_server

_log = get_logger("fedotmas.mcp.discovery")


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


def discover_servers(
    base_dir: str | Path = "mcp-servers",
) -> dict[str, MCPServerConfig]:
    """Scan *base_dir*/\\*/pyproject.toml for ``[tool.fedotmas.mcp]`` sections.

    Each discovered server is built via :func:`directory_server` so it launches
    with ``uv run --directory <server_dir> <entry_point>``.
    """
    repo_root = _find_repo_root()
    servers_dir = repo_root / base_dir

    if not servers_dir.is_dir():
        _log.warning("MCP servers directory not found: {}", servers_dir)
        return {}

    result: dict[str, MCPServerConfig] = {}

    for pyproject_path in sorted(servers_dir.glob("*/pyproject.toml")):
        server_dir = pyproject_path.parent
        try:
            data = tomllib.loads(pyproject_path.read_text())
        except Exception:
            _log.warning("Failed to parse {}", pyproject_path)
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

        # Use relative path from repo root for portability
        relative_dir = str(server_dir.relative_to(repo_root))

        result[name] = directory_server(
            directory=relative_dir,
            entry_point=entry_point,
            description=description,
            tags=tags,
        )
        _log.debug("Discovered MCP server: {} -> {}", name, relative_dir)

    return result
