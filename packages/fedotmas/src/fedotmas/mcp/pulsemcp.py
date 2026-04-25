"""MCP server discovery via Smithery registry (https://smithery.ai)."""
from __future__ import annotations

import asyncio
import re
from urllib.parse import quote

import httpx

from fedotmas.common.logging import get_logger
from fedotmas.mcp._config import HttpMCPServer, MCPServerConfig

_log = get_logger("fedotmas.mcp.pulsemcp")

_SMITHERY_SEARCH = "https://registry.smithery.ai/servers"
_SMITHERY_DETAIL = "https://registry.smithery.ai/servers/{name}"
_TIMEOUT = 10.0


async def discover_relevant_servers(
    task: str,
    *,
    count: int = 5,
) -> dict[str, MCPServerConfig]:
    """Search Smithery registry for MCP servers relevant to *task*.

    Returns ``{slug: HttpMCPServer}`` for deployed remote servers.
    Always returns an empty dict on any error so it never breaks the meta-agent flow.
    """
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            candidates = await _search(client, task, count * 2)
            deployed = [s for s in candidates if s.get("isDeployed")][:count]
            details = await asyncio.gather(
                *[_fetch_detail(client, s["qualifiedName"]) for s in deployed],
                return_exceptions=True,
            )
    except Exception as exc:
        _log.warning("Smithery discovery failed ({}): {}", type(exc).__name__, exc)
        return {}

    result: dict[str, MCPServerConfig] = {}
    for server, detail in zip(deployed, details):
        if isinstance(detail, Exception):
            _log.debug("Detail fetch failed for {}: {}", server["qualifiedName"], detail)
            continue
        url: str = detail.get("deploymentUrl", "")
        if not url:
            continue
        slug = _slugify(server["qualifiedName"])
        raw_desc = server.get("description") or f"MCP server: {server['displayName']}"
        description = raw_desc[:150].rstrip() + ("…" if len(raw_desc) > 150 else "")
        result[slug] = HttpMCPServer(
            url=url,
            description=description,
            tags=("smithery", "discovered"),
        )
        _log.debug("Smithery: discovered {} -> {}", slug, url)

    _log.info(
        "Smithery discovery: {}/{} deployed servers for task",
        len(result),
        len(deployed),
    )
    return result


async def _search(client: httpx.AsyncClient, query: str, page_size: int) -> list[dict]:
    resp = await client.get(
        _SMITHERY_SEARCH,
        params={"q": query, "pageSize": page_size},
    )
    resp.raise_for_status()
    return resp.json().get("servers", [])


async def _fetch_detail(client: httpx.AsyncClient, qualified_name: str) -> dict:
    url = _SMITHERY_DETAIL.format(name=quote(qualified_name, safe=""))
    resp = await client.get(url)
    resp.raise_for_status()
    return resp.json()


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug[:64]
