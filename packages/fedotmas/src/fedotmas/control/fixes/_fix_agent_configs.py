"""Apply a coherent tool/contract repair as one validated transaction."""

from __future__ import annotations

import json

from google.adk.tools import ToolContext
from pydantic import BaseModel, ConfigDict

from fedotmas.control.fixes._guardrails import run_config_guardrails
from fedotmas.maw.contracts import handoff_contract_errors
from fedotmas.maw.models import ArtifactContract, ArtifactRequirement, MAWConfig

AVAILABLE_TOOLS_STATE_KEY = "_fedotmas_recovery_available_tools"
_EDITABLE_FIELDS = {"instruction", "tools", "output_contract", "input_requirements"}


class AgentConfigRepair(BaseModel):
    """Only explicitly supplied fields replace the corresponding agent fields."""

    model_config = ConfigDict(extra="forbid")

    agent_name: str
    instruction: str | None = None
    tools: list[str] | None = None
    output_contract: ArtifactContract | None = None
    input_requirements: list[ArtifactRequirement] | None = None


async def repair_agent_configs(
    tool_context: ToolContext, updates: list[AgentConfigRepair], reasoning: str
) -> str:
    """Atomically repair instructions, tools and contracts of related agents.

    Pass updates as an array of objects, NOT a JSON-encoded string. Omit fields
    that should stay unchanged. Preserve state references such as {source?} in
    instructions. Include producer and consumer together for handoff changes.
    Names, output keys, models, pipeline and required dependencies cannot change.
    """
    try:
        payload = [
            AgentConfigRepair.model_validate(update).model_dump(exclude_unset=True)
            for update in updates
        ]
    except (ValueError, TypeError) as exc:
        return f"Error applying repair (configuration unchanged): {exc}"
    # Keep the existing entry point and validation semantics for API callers.
    return await fix_agent_configs(tool_context, json.dumps(payload), reasoning)


async def fix_agent_configs(
    tool_context: ToolContext, updates_json: str, reasoning: str
) -> str:
    """Repair tools or handoff contracts for related agents atomically.

    updates_json is a JSON array of objects with agent_name and the fields to
    replace: instruction, tools (MCP server names), output_contract, and/or
    input_requirements. Include BOTH producer and consumer when fixing a shape
    mismatch. Use full dotted paths, including [] for list items. Preserve the
    information required by the task; never suppress a failure by deleting its
    semantic requirements. Agent names, output keys, models, and the pipeline
    cannot change. Nothing is applied unless the entire repair validates.
    """
    try:
        updates = json.loads(updates_json)
        if not isinstance(updates, list) or not updates:
            raise ValueError("updates_json must be a non-empty JSON array")
        raw = tool_context.state["config"]
        config = (
            MAWConfig.model_validate_json(raw)
            if isinstance(raw, str)
            else MAWConfig.model_validate(raw)
        )
        payload = config.model_dump()
        agents = {agent["name"]: agent for agent in payload["agents"]}
        available = tool_context.state.get(AVAILABLE_TOOLS_STATE_KEY)
        touched: set[str] = set()
        for update in updates:
            if not isinstance(update, dict):
                raise TypeError("Every update must be an object")
            name = update.get("agent_name")
            if not isinstance(name, str) or name not in agents or name in touched:
                raise ValueError(f"Unknown or duplicate agent_name: {name!r}")
            fields = set(update) - {"agent_name"}
            if not fields or fields - _EDITABLE_FIELDS:
                raise ValueError(
                    f"Only these fields may change: {sorted(_EDITABLE_FIELDS)}"
                )
            if "tools" in fields:
                if not isinstance(available, list):
                    raise ValueError(
                        "Available MCP registry is missing; cannot authorize tool changes"
                    )
                selected = update["tools"]
                if not isinstance(selected, list) or not all(
                    isinstance(t, str) for t in selected
                ):
                    raise ValueError("tools must be a list of MCP server names")
                unknown = set(selected) - set(available)
                if unknown:
                    raise ValueError(
                        f"Unavailable MCP servers: {sorted(unknown)}. Available: {available}"
                    )
            if "input_requirements" in fields:
                # Changing field paths is legitimate; deleting a dependency to
                # manufacture a completed status is not a recovery operation.
                proposed = update["input_requirements"]
                if not isinstance(proposed, list) or not all(
                    isinstance(r, dict) for r in proposed
                ):
                    raise ValueError("input_requirements must be a list of objects")
                for previous in agents[name]["input_requirements"]:
                    replacements = [
                        r
                        for r in proposed
                        if r.get("source_key") == previous["source_key"]
                    ]
                    if not replacements:
                        raise ValueError(
                            "Repair must preserve existing handoff dependencies"
                        )
                    for kind in ("required_fields", "identity_fields"):
                        if previous[kind] and not any(
                            r.get(kind) for r in replacements
                        ):
                            raise ValueError(
                                f"Repair must preserve non-empty {kind} requirements"
                            )
            agents[name].update({key: update[key] for key in fields})
            touched.add(name)
        candidate = MAWConfig.model_validate(payload)
        errors = [
            *run_config_guardrails(candidate),
            *handoff_contract_errors(candidate),
        ]
        if errors:
            raise ValueError("; ".join(errors))
        if candidate == config:
            raise ValueError("The update did not change the configuration")
    except (ValueError, TypeError, KeyError) as exc:
        return f"Error applying repair (configuration unchanged): {exc}"
    tool_context.state["config"] = candidate.model_dump_json()
    return f"Updated agents {sorted(touched)}. Reason: {reasoning}"
