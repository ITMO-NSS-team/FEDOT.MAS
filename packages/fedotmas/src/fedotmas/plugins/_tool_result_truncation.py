from __future__ import annotations

from typing import Any, Optional

from google.adk.plugins import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext

from fedotmas.common.logging import get_logger

_log = get_logger("fedotmas.plugins.tool_result_truncation")


class ToolResultTruncationPlugin(BasePlugin):
    """Cap oversized tool result strings before they enter model context."""

    def __init__(
        self,
        *,
        max_string_chars: int = 50000,
        name: str = "fedotmas_tool_result_truncation",
    ) -> None:
        if max_string_chars < 1:
            raise ValueError("max_string_chars must be >= 1")
        super().__init__(name=name)
        self.max_string_chars = max_string_chars

    async def after_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        result: dict,
    ) -> Optional[dict]:
        truncated, changed = _truncate_value(result, self.max_string_chars)
        if not changed:
            return None

        agent_name = tool_context._invocation_context.agent.name  # noqa: E501  # ty: ignore[unresolved-attribute]
        _log.warning(
            "Tool result truncated | agent={} tool={} max_string_chars={}",
            agent_name,
            tool.name,
            self.max_string_chars,
        )
        if not isinstance(truncated, dict):
            truncated = {"result": truncated}
        return {
            **truncated,
            "truncated": True,
            "complete": False,
            "max_chars": self.max_string_chars,
            "recommended_next_action": (
                "Use targeted find, section extraction, table extraction, or chunked "
                "read before giving a final answer."
            ),
        }


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
