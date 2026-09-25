from __future__ import annotations

import json
import math
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

        latest_state_response = next(
            (
                index
                for index in range(len(responses) - 1, -1, -1)
                if _has_research_state(responses[index].response)
            ),
            None,
        )
        # Compact older evidence but retain the latest controller snapshot.
        for index, response in enumerate(responses[:-1]):
            if index != latest_state_response:
                response.response = _compact_metadata(response.response)
        # Reserve most active context for the newest payload while keeping some
        # compact source metadata for older evidence.
        metadata_limit = max(1, self.max_agent_total_chars // 3)
        for index, response in enumerate(responses[:-1]):
            older_chars = sum(
                len(json.dumps(item.response, ensure_ascii=False, default=str))
                for item in responses[:-1]
            )
            if older_chars <= metadata_limit:
                break
            if index != latest_state_response:
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
        for index, response in enumerate(responses[:-1]):
            if active_chars() <= self.max_agent_total_chars:
                break
            if index != latest_state_response:
                response.response = {}
        if active_chars() > self.max_agent_total_chars and latest_state_response is not None:
            # The newest controller snapshot outranks ordinary tool evidence. Compact
            # that snapshot by schema, then spend any remaining room on the newest
            # ordinary response. Never leave a pinned snapshot over the context cap.
            state_response = responses[latest_state_response]
            newest_response = (
                deepcopy(responses[-1].response)
                if latest_state_response != len(responses) - 1
                else None
            )
            for index, response in enumerate(responses):
                if index != latest_state_response:
                    response.response = {}
            state_response.response, _ = _truncate_total(
                state_response.response,
                self.max_agent_total_chars - 2 * (len(responses) - 1),
            )
            if newest_response is not None:
                state_chars = len(
                    json.dumps(state_response.response, ensure_ascii=False, default=str)
                )
                allowance = (
                    self.max_agent_total_chars
                    - state_chars
                    - 2 * (len(responses) - 2)
                )
                if allowance > 2:
                    responses[-1].response, _ = _truncate_total(
                        newest_response, allowance
                    )

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
            if str(key).casefold() == "research_state":
                result[key] = deepcopy(item)
                continue
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
    if _has_research_state(result):
        _compact_research_state_slots(result, limit)
        changed = True
    while len(json.dumps(result, ensure_ascii=False, default=str)) > limit:
        lists = _ordinary_containers(result, list)
        nonempty = [items for items in lists if len(items) > 1]
        if nonempty:
            # Keep the newest list entry where possible.
            max(nonempty, key=lambda items: len(json.dumps(items, default=str))).pop(0)
            changed = True
            continue
        strings = _ordinary_string_slots(result)
        if not strings:
            dictionaries = [
                item
                for item in _ordinary_containers(result, dict)
                if any(str(key).casefold() != "research_state" for key in item)
            ]
            if not dictionaries:
                if _has_research_state(result):
                    _compact_research_state_slots(result, limit)
                    changed = True
                    continue
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
            removable = [
                key
                for key in container
                if str(key).lower() not in keep
                and str(key).casefold() != "research_state"
            ]
            if not removable:
                removable = [
                    key
                    for key in container
                    if str(key).casefold() != "research_state"
                ]
                if not removable and _has_research_state(result):
                    return result, True
                if not removable:
                    removable = list(container)
            del container[removable[-1]]
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


def _has_research_state(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            str(key).casefold() == "research_state" or _has_research_state(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_has_research_state(item) for item in value)
    return False


def _compact_research_state_slots(value: Any, limit: int) -> None:
    root = value

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            for key, item in list(node.items()):
                if str(key).casefold() == "research_state":
                    original = node[key]
                    node[key] = {}
                    overhead = len(json.dumps(root, ensure_ascii=False, default=str))
                    available = limit - overhead + 2
                    if available > 0:
                        node[key] = _compact_research_state(original, available)
                    else:
                        node[key] = original
                else:
                    visit(item)
        elif isinstance(node, list):
            for item in node:
                visit(item)

    visit(root)


def _compact_research_state(value: Any, limit: int) -> dict[str, Any]:
    """Keep controller continuation fields valid while dropping old history."""
    if not isinstance(value, dict):
        raise TypeError("research_state must be an object to preserve continuation state")

    def bounded_text(item: Any, fallback: str = "") -> str:
        if not isinstance(item, str):
            return fallback
        return " ".join(item.split())[:240]

    def strings(item: Any) -> list[str]:
        if not isinstance(item, list):
            return []
        seen = set()
        result = []
        for entry in item:
            cleaned = bounded_text(entry)
            key = cleaned.casefold()
            if cleaned and key not in seen:
                result.append(cleaned)
                seen.add(key)
        return result[-30:]

    def count(item: Any) -> int:
        return item if isinstance(item, int) and not isinstance(item, bool) and item >= 0 else 0

    raw_intents = value.get("search_intents")
    search_intents = []
    if isinstance(raw_intents, list):
        for profile in raw_intents:
            if not isinstance(profile, dict):
                continue
            terms = strings(profile.get("terms"))
            if terms:
                search_intents.append(
                    {"terms": terms, "entities": strings(profile.get("entities"))}
                )

    pending = value.get("telemetry")
    pending = pending if isinstance(pending, dict) else {}
    recommendation = pending.get("pending_recommendation")
    if isinstance(recommendation, dict):
        action = recommendation.get("action")
        recommendation = (
            {
                "id": count(recommendation.get("id")),
                "action": action,
                "searches_before": count(recommendation.get("searches_before")),
            }
            if count(recommendation.get("id")) > 0
            and action in {"continue_search", "change_strategy", "strategy_blocked", "synthesize"}
            else None
        )
    else:
        recommendation = None

    result: dict[str, Any] = {
        "version": 1,
        "research_id": bounded_text(value.get("research_id"), "default") or "default",
        "goal": bounded_text(value.get("goal"), "[goal unavailable]") or "[goal unavailable]",
        "unresolved_questions": strings(value.get("unresolved_questions")),
        "required_fields": strings(value.get("required_fields")),
        "filled_fields": strings(value.get("filled_fields")),
        "search_count": count(value.get("search_count")),
        "failed_attempt_count": count(value.get("failed_attempt_count")),
        "failed_strategy_counts": {
            bounded_text(key): count(number)
            for key, number in (value.get("failed_strategy_counts") or {}).items()
            if isinstance(key, str) and bounded_text(key)
        } if isinstance(value.get("failed_strategy_counts"), dict) else {},
        "search_intents": search_intents[-2:],
        "remaining_budget": value.get("remaining_budget")
        if isinstance(value.get("remaining_budget"), int | float)
        and not isinstance(value.get("remaining_budget"), bool)
        and math.isfinite(value["remaining_budget"])
        and value["remaining_budget"] >= 0 else None,
        "confidence": value.get("confidence")
        if isinstance(value.get("confidence"), int | float)
        and not isinstance(value.get("confidence"), bool)
        and math.isfinite(value["confidence"])
        and 0 <= value["confidence"] <= 1 else None,
        "telemetry": {
            "controller_calls": count(pending.get("controller_calls")),
            "recommendation_count": max(
                count(pending.get("recommendation_count")),
                count(recommendation.get("id")) if recommendation else 0,
            ),
            "strategy_blocked_events": count(pending.get("strategy_blocked_events")),
            "followed_recommendations": count(pending.get("followed_recommendations")),
            "unfollowed_recommendations": count(pending.get("unfollowed_recommendations")),
            "unknown_follow_through": count(pending.get("unknown_follow_through")),
            "pending_recommendation": recommendation,
        },
    }
    # These fields can be useful to the next recommendation, but history and old
    # evidence are evicted before unresolved work or identity/counters.
    optional = (
        ("evidence_urls", strings(value.get("evidence_urls"))[-3:]),
        ("independent_sources", strings(value.get("independent_sources"))[-3:]),
        ("findings", strings(value.get("findings"))[-3:]),
        ("evidence", strings(value.get("evidence"))[-3:]),
    )
    for key, entries in optional:
        if entries:
            result[key] = entries
    shrinkable = ("search_intents", "findings", "evidence", "evidence_urls", "independent_sources")
    while len(json.dumps(result, ensure_ascii=False, default=str)) > limit:
        shrunk = False
        for field in shrinkable:
            entries = result.get(field)
            if isinstance(entries, list) and entries:
                entries.pop(0)
                if not entries:
                    result.pop(field, None)
                shrunk = True
                break
        if not shrunk:
            raise ValueError(
                "context character limit is too small for valid research_state continuation"
            )
    return result


def _ordinary_containers(value: Any, kind: type) -> list[Any]:
    found = [value] if isinstance(value, kind) else []
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).casefold() != "research_state":
                found.extend(_ordinary_containers(item, kind))
    elif isinstance(value, list):
        for item in value:
            found.extend(_ordinary_containers(item, kind))
    return found


def _ordinary_string_slots(value: Any) -> list[tuple[Any, Any]]:
    found: list[tuple[Any, Any]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).casefold() == "research_state":
                continue
            if isinstance(item, str):
                found.append((value, key))
            else:
                found.extend(_ordinary_string_slots(item))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            if isinstance(item, str):
                found.append((value, index))
            else:
                found.extend(_ordinary_string_slots(item))
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
