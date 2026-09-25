from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from google.adk.plugins import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext

from fedotmas.common.logging import get_logger
from fedotmas.mcp import strip_tool_name_prefix

_log = get_logger("fedotmas.plugins.tool_result_truncation")


class ToolResultTruncationPlugin(BasePlugin):
    """Cap oversized tool result strings before they enter model context."""

    def __init__(
        self,
        *,
        max_string_chars: int = 50000,
        max_total_chars: int | None = None,
        aggregate_tool_names: set[str] | None = None,
        max_agent_total_chars: int | None = None,
        name: str = "fedotmas_tool_result_truncation",
    ) -> None:
        if max_string_chars < 1:
            raise ValueError("max_string_chars must be >= 1")
        if max_total_chars is not None and max_total_chars < 500:
            raise ValueError("max_total_chars must be >= 500")
        if max_agent_total_chars is not None and max_agent_total_chars < 500:
            raise ValueError("max_agent_total_chars must be >= 500")
        super().__init__(name=name)
        self.max_string_chars = max_string_chars
        self.max_total_chars = max_total_chars
        self.aggregate_tool_names = {
            name.lower() for name in (aggregate_tool_names or set())
        }
        self.max_agent_total_chars = max_agent_total_chars
        self._agent_chars: dict[tuple[str, str], int] = {}

    async def before_run_callback(self, *, invocation_context) -> None:
        session_id = invocation_context.session.id
        self._agent_chars = {
            key: chars
            for key, chars in self._agent_chars.items()
            if key[0] != session_id
        }

    async def after_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        result: dict,
    ) -> dict | None:
        truncated, changed = _truncate_value(result, self.max_string_chars)
        total_limit = self.max_total_chars
        agent_name = tool_context._invocation_context.agent.name  # ty: ignore[unresolved-attribute]
        session_id = tool_context._invocation_context.session.id
        aggregate = (
            "*" in self.aggregate_tool_names
            or strip_tool_name_prefix(tool.name).lower() in self.aggregate_tool_names
            or (
                self.max_agent_total_chars is not None and not self.aggregate_tool_names
            )
        )
        agent_key = (session_id, agent_name)
        remaining_agent = (
            self.max_agent_total_chars - self._agent_chars.get(agent_key, 0)
            if self.max_agent_total_chars is not None
            else None
        )
        if aggregate and (total_limit is not None or remaining_agent is not None):
            per_result_limit = total_limit or self.max_string_chars
            if remaining_agent is not None:
                per_result_limit = min(per_result_limit, max(0, remaining_agent - 250))
            if per_result_limit >= 500:
                truncated, total_changed = _truncate_total(
                    truncated, per_result_limit - 250
                )
            elif remaining_agent is not None:
                truncated = _compact_metadata(truncated)
                truncated, _ = _truncate_total(
                    truncated, min(self.max_string_chars, max(250, remaining_agent))
                )
                total_changed = True
            changed = changed or total_changed
        if not changed and remaining_agent is None:
            return None
        if not changed:
            if aggregate and self.max_agent_total_chars is not None:
                serialized_size = len(
                    json.dumps(truncated, ensure_ascii=False, default=str)
                )
                self._agent_chars[agent_key] = min(
                    self.max_agent_total_chars,
                    self._agent_chars.get(agent_key, 0) + serialized_size,
                )
            return None
        _log.warning(
            "Tool result truncated | agent={} tool={} max_string_chars={}",
            agent_name,
            tool.name,
            self.max_string_chars,
        )
        if not isinstance(truncated, dict):
            truncated = {"result": truncated}
        compact_result = {
            **truncated,
            "truncated": True,
            "complete": False,
            "max_chars": self.max_string_chars,
            "max_total_chars": total_limit,
            "recommended_next_action": (
                "Use targeted find, section extraction, table extraction, or chunked "
                "read before giving a final answer."
            ),
            **(
                {"agent_evidence_char_budget": self.max_agent_total_chars}
                if self.max_agent_total_chars is not None
                else {}
            ),
        }
        if aggregate and self.max_agent_total_chars is not None:
            serialized_size = len(
                json.dumps(compact_result, ensure_ascii=False, default=str)
            )
            self._agent_chars[agent_key] = min(
                self.max_agent_total_chars,
                self._agent_chars.get(agent_key, 0) + serialized_size,
            )
        return compact_result


def _truncate_value(value: Any, max_chars: int) -> tuple[Any, bool]:
    if isinstance(value, str):
        if len(value) <= max_chars:
            return value, False
        suffix = f"\n\n... (truncated to {max_chars}/{len(value)} chars)"
        return value[:max_chars] + suffix, True

    if isinstance(value, list):
        changed = False
        items = []
        for item in value:
            truncated, item_changed = _truncate_value(item, max_chars)
            items.append(truncated)
            changed = changed or item_changed
        return items, changed

    if isinstance(value, dict):
        changed = False
        result = {}
        for key, item in value.items():
            truncated, item_changed = _truncate_value(item, max_chars)
            result[key] = truncated
            changed = changed or item_changed
        return result, changed

    return value, False


def _truncate_total(value: Any, limit: int) -> tuple[Any, bool]:
    """Bound serialized nested content, including many individually short entries."""
    if len(json.dumps(value, ensure_ascii=False, default=str)) <= limit:
        return value, False
    result = deepcopy(value)
    changed = False
    while len(json.dumps(result, ensure_ascii=False, default=str)) > limit:
        lists = _containers(result, list)
        nonempty = [items for items in lists if items]
        if nonempty:
            max(nonempty, key=lambda items: len(json.dumps(items, default=str))).pop()
            changed = True
            continue
        strings = _string_slots(result)
        if not strings:
            return {}, True
        container, key = max(strings, key=lambda slot: len(slot[0][slot[1]]))
        excess = len(json.dumps(result, ensure_ascii=False, default=str)) - limit
        old = container[key]
        container[key] = old[: max(0, len(old) - excess - 20)]
        changed = True
    return result, changed


def _containers(value: Any, kind: type) -> list[Any]:
    found = [value] if isinstance(value, kind) else []
    if isinstance(value, dict):
        for item in value.values():
            found.extend(_containers(item, kind))
    elif isinstance(value, list):
        for item in value:
            found.extend(_containers(item, kind))
    return found


def _string_slots(value: Any) -> list[tuple[Any, Any]]:
    found: list[tuple[Any, Any]] = []
    if isinstance(value, (dict, list)):
        items = value.items() if isinstance(value, dict) else enumerate(value)
        for key, item in items:
            if isinstance(item, str):
                found.append((value, key))
            else:
                found.extend(_string_slots(item))
    return found


def _compact_metadata(value: Any) -> dict[str, Any]:
    """Keep source and evidence metadata when this agent's text budget is spent."""
    keep = {
        "url",
        "source",
        "source_id",
        "id",
        "title",
        "query",
        "status",
        "error",
        "error_code",
        "result_count",
        "total_results",
        "snippet",
        "excerpt",
        "summary",
        "evidence",
        "value",
    }

    def compact(item: Any) -> Any:
        if isinstance(item, dict):
            selected: dict[str, Any] = {}
            for key, nested in item.items():
                if key.lower() in keep and isinstance(
                    nested, (str, int, float, bool, type(None))
                ):
                    selected[key] = nested[:400] if isinstance(nested, str) else nested
                elif isinstance(nested, dict):
                    nested_selected = compact(nested)
                    if nested_selected:
                        selected[key] = nested_selected
                elif isinstance(nested, list):
                    nested_selected = [
                        compact(value)
                        for value in nested[:3]
                        if isinstance(value, dict | list)
                    ]
                    nested_selected = [value for value in nested_selected if value]
                    if nested_selected:
                        selected[key] = nested_selected
            return selected
        if isinstance(item, list):
            return [
                compact(value) for value in item[:3] if isinstance(value, dict | list)
            ]
        return {}

    return (
        compact(value)
        if isinstance(value, dict)
        else {"result_type": type(value).__name__}
    )
