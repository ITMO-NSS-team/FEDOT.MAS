"""Describe MCP servers the workspace did not declare.

A local server carries a hand-written description from its ``pyproject.toml``.
A server handed in by URL carries none, and :func:`get_server_descriptions`
then advertises it to the meta-agent as ``"MCP server: <name>"`` -- which names
it without saying what it does, so the meta-agent connects to it and never
picks it.  Asking the server for its own tool list is the only description
available, and the same connection proves the address works before a run pays
for generation.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace

from google.adk.tools.base_tool import BaseTool

from fedotmas.common.logging import get_logger
from fedotmas.mcp._config import MCPServerConfig
from fedotmas.mcp.registry import list_server_tools

_log = get_logger("fedotmas.mcp.describe")

#: Character budget for the whole generated description.  The meta-agent picks
#: a *server*, so the listing only has to convey what kind of server it is, and
#: one entry must not crowd the task out of the prompt: lightpanda's 32 tools
#: run past 1100 characters unbudgeted.  Tools are listed until the next would
#: exceed this, and the remainder becomes a count.
MAX_DESCRIPTION_CHARS = 400

#: Per-tool description budget.  Server authors write paragraphs (lightpanda's
#: `extract` runs to 1600 characters); the first sentence is what distinguishes
#: one tool from another.
MAX_TOOL_DESCRIPTION_CHARS = 120


@dataclass(frozen=True)
class UnreachableServer:
    """A server that did not answer a tool listing."""

    name: str
    reason: str


def build_description(name: str, tools: list[BaseTool]) -> str:
    """Render *tools* as a one-line description of the server."""
    if not tools:
        return f"MCP server '{name}', which advertises no tools."

    parts: list[str] = []
    used = len("Tools: .")
    for tool in tools:
        summary = _first_sentence(tool.description or "")
        part = f"{tool.name} ({summary})" if summary else tool.name
        # Always list the first tool: a server whose one tool has a long
        # description would otherwise be described as nothing but a count.
        if parts and used + len(part) + 2 > MAX_DESCRIPTION_CHARS:
            break
        parts.append(part)
        used += len(part) + 2

    remainder = len(tools) - len(parts)
    tail = f", and {remainder} more" if remainder else ""
    return f"Tools: {', '.join(parts)}{tail}."


async def describe_servers(
    registry: dict[str, MCPServerConfig],
) -> tuple[dict[str, MCPServerConfig], list[UnreachableServer]]:
    """Fill in missing descriptions by asking each server for its tools.

    Returns the registry with descriptions added and the servers that did not
    answer.  Servers that already carry a description are neither probed nor
    reported: a declared description is the author's, and re-deriving it would
    spend a connection to replace better text with worse.
    """
    pending = [name for name, cfg in registry.items() if not cfg.description]
    if not pending:
        return registry, []

    _log.debug("Describing MCP servers without a description: {}", pending)
    listings = await asyncio.gather(
        *(list_server_tools(name, registry) for name in pending),
        return_exceptions=True,
    )

    described = dict(registry)
    unreachable: list[UnreachableServer] = []
    for name, listing in zip(pending, listings, strict=True):
        if isinstance(listing, BaseException):
            reason = _reason(listing)
            _log.warning("MCP server '{}' did not answer: {}", name, reason)
            unreachable.append(UnreachableServer(name, reason))
            continue
        description = build_description(name, listing)
        described[name] = replace(registry[name], description=description)
        _log.info("Described MCP server '{}' | {}", name, description)

    return described, unreachable


def _first_sentence(text: str) -> str:
    text = " ".join(text.split())
    head = text.split(". ")[0].rstrip(".")
    if len(head) > MAX_TOOL_DESCRIPTION_CHARS:
        head = head[: MAX_TOOL_DESCRIPTION_CHARS - 1].rstrip() + "…"
    return head


def _reason(error: BaseException) -> str:
    if isinstance(error, TimeoutError):
        return "timed out"
    message = " ".join(str(error).split())
    return f"{type(error).__name__}: {message}" if message else type(error).__name__
