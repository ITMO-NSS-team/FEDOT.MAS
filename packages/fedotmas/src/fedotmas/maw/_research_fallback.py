"""Bounded alternative retrieval after evidence stagnation (no extra budget)."""

import hashlib
from typing import Any

from fedotmas.mcp.capabilities import (
    ToolCapability,
    normalize_tool_name,
    tool_capability,
)
from fedotmas.plugins._research_telemetry import _sanitize_url

RETRIEVAL_STATE_KEY = "__fedotmas_retrieval_attempts"
MAX_FALLBACK_ROUTES = 2
MAX_CALLS_PER_ROUTE = 2
_NAVIGATION_METHODS = {"web:goto", "web:navigate"}


def _source_id(value: Any) -> str | None:
    normalized = _sanitize_url(value)
    return hashlib.sha256(normalized.encode()).hexdigest() if normalized else None


def _family(name: str) -> str | None:
    if normalize_tool_name(name) in {"download", "download_file"}:
        return "download"
    return {
        ToolCapability.URL_INSPECTION: "web",
        ToolCapability.DOCUMENT_INSPECTION: "document",
        ToolCapability.MEDIA_INSPECTION: "media",
    }.get(tool_capability(name))


def _target(args: dict) -> str | None:
    value = (
        args.get("url") or args.get("file_path") or args.get("path") or args.get("urls")
    )
    if isinstance(value, list):
        value = value[0] if len(value) == 1 else None
    return _source_id(value)


def record_retrieval(state: Any, agent: str, tool: str, args: dict) -> None:
    """Remember executed routes, including failures, but not policy blocks."""
    root = dict(state.get(RETRIEVAL_STATE_KEY) or {})
    entry = dict(root.get(agent) or {})
    source, family = _target(args), _family(tool)
    if (
        source is None
        and family == "web"
        and not any(key in args for key in ("url", "urls", "path", "file_path"))
    ):
        source = entry.get("current_web_source")
    if source is None or family is None:
        return
    attempts = dict(entry.get("attempts") or {})
    if source not in attempts and len(attempts) >= 64:
        return
    methods = list(attempts.get(source) or [])
    method = family + ":" + normalize_tool_name(tool)
    if method not in methods:
        methods.append(method)
    attempts[source] = methods
    entry["attempts"] = attempts
    if method in _NAVIGATION_METHODS:
        entry["current_web_source"] = source
    root[agent] = entry
    state[RETRIEVAL_STATE_KEY] = root


def allow_fallback(
    state: Any,
    agent: str,
    tool: str,
    candidates: list[dict],
    *,
    args: dict | None = None,
    reserve: bool = False,
) -> bool:
    """Allow two methods on each of two new candidate/backend routes.

    A changed selector/query is not a new method. Unknown sources do not bypass
    convergence. Budgets persist in session state and are never reset by progress.
    With no args this is a read-only check for model tool visibility.
    """
    family = _family(tool)
    if family is None:
        return False
    root = dict(state.get(RETRIEVAL_STATE_KEY) or {})
    entry = dict(root.get(agent) or {})
    attempts = entry.get("attempts") or {}
    routes = dict(entry.get("fallback_routes") or {})
    known = {value for c in candidates if (value := _source_id(c.get("url")))}
    targets = [value] if args is not None and (value := _target(args)) else []
    # DOM readers often operate on the current page without a URL argument.
    if (
        args is not None
        and not targets
        and family == "web"
        and not any(key in args for key in ("url", "urls", "path", "file_path"))
    ):
        current = entry.get("current_web_source")
        if current:
            targets = [current]
    if args is None:
        targets = list(known | set(attempts))
    method = family + ":" + normalize_tool_name(tool)
    for target in targets:
        route = target + ":" + family
        used = list(routes.get(route) or [])
        previous = attempts.get(target) or []
        family_methods = {m for m in previous if m.startswith(family + ":")}
        if method in used or method in previous:
            continue
        if route in routes:
            eligible = len(used) < MAX_CALLS_PER_ROUTE
        else:
            eligible = len(routes) < MAX_FALLBACK_ROUTES and (
                (target in known and not previous)
                or (
                    bool(previous)
                    and (
                        not family_methods
                        or (
                            family_methods <= _NAVIGATION_METHODS
                            and method not in _NAVIGATION_METHODS
                        )
                    )
                )
            )
        if not eligible:
            continue
        if reserve:
            routes[route] = [*used, method]
            entry["fallback_routes"] = routes
            root[agent] = entry
            state[RETRIEVAL_STATE_KEY] = root
        return True
    return False
