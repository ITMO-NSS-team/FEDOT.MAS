from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
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

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> None:
        """Roll older tool evidence out of active context while retaining metadata."""
        del callback_context
        if self.max_agent_total_chars is None:
            return
        responses = [
            part.function_response
            for content in llm_request.contents
            for part in content.parts or []
            if part.function_response is not None
            and isinstance(part.function_response.response, dict)
        ]
        if not responses:
            return

        def active_chars() -> int:
            return sum(
                len(json.dumps(item.response, ensure_ascii=False, default=str))
                for item in responses
            )

        if active_chars() <= self.max_agent_total_chars:
            return

        # Compact older responses, preserving source identifiers and URLs.
        for response in responses[:-1]:
            response.response = _compact_metadata(response.response)
        # Reserve most active context for the newest payload while keeping some
        # compact source metadata for older evidence.
        metadata_limit = max(1, self.max_agent_total_chars // 3)
        for response in responses[:-1]:
            older_chars = sum(
                len(json.dumps(item.response, ensure_ascii=False, default=str))
                for item in responses[:-1]
            )
            if older_chars <= metadata_limit:
                break
            response.response = {}
        newest = responses[-1]
        if active_chars() > self.max_agent_total_chars:
            older_chars = active_chars() - len(
                json.dumps(newest.response, ensure_ascii=False, default=str)
            )
            newest.response, _ = _truncate_total(
                newest.response,
                max(1, self.max_agent_total_chars - older_chars - 10),
            )
        for response in responses[:-1]:
            if active_chars() <= self.max_agent_total_chars:
                break
            response.response = {}

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
        aggregate = (
            "*" in self.aggregate_tool_names
            or strip_tool_name_prefix(tool.name).lower() in self.aggregate_tool_names
            or (
                self.max_agent_total_chars is not None and not self.aggregate_tool_names
            )
        )
        if aggregate:
            result_limit = total_limit or self.max_string_chars
            truncated, total_changed = _truncate_total(
                truncated, max(1, result_limit - 250)
            )
            changed = changed or total_changed
        if not changed:
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
        }
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
        nonempty = [items for items in lists if len(items) > 1]
        if nonempty:
            # Keep the newest list entry where possible.
            max(nonempty, key=lambda items: len(json.dumps(items, default=str))).pop(0)
            changed = True
            continue
        strings = _string_slots(result)
        if not strings:
            dictionaries = [item for item in _containers(result, dict) if item]
            if not dictionaries:
                return (0 if limit == 1 else {}), True
            container = max(
                dictionaries,
                key=lambda item: len(json.dumps(item, ensure_ascii=False, default=str)),
            )
            keep = {
                "value",
                "result",
                "content",
                "stdout",
                "output",
                "evidence",
                "data",
                "snippet",
                "text",
            }
            removable = [key for key in container if str(key).lower() not in keep]
            del container[removable[-1] if removable else next(reversed(container))]
            changed = True
            continue
        container, key = max(strings, key=lambda slot: len(slot[0][slot[1]]))
        excess = len(json.dumps(result, ensure_ascii=False, default=str)) - limit
        old = container[key]
        shortened = old[: max(0, len(old) - excess - 20)]
        if shortened == old:
            # Empty strings can still sit under an oversized structural envelope.
            del container[key]
        else:
            container[key] = shortened
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
        "uri",
        "source_url",
        "source_urls",
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
                elif key.lower() in keep and isinstance(nested, list):
                    selected[key] = [
                        item[:300] if isinstance(item, str) else item
                        for item in nested[:3]
                        if isinstance(item, str | int | float | bool | type(None))
                    ]
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
