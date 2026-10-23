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


class OutputFieldRename(BaseModel):
    """An explicit relocation of an existing output obligation."""

    model_config = ConfigDict(extra="forbid")

    old_path: str
    new_path: str


class InputFieldRename(OutputFieldRename):
    """An explicit relocation of an input obligation from one source."""

    source_key: str


class AgentConfigRepair(BaseModel):
    """Only supplied non-null fields replace the corresponding agent fields.

    Omitted or null fields mean no change. Empty lists are explicit edits and
    still pass through dependency and output-contract preservation checks.
    """

    model_config = ConfigDict(extra="forbid")

    agent_name: str
    instruction: str | None = None
    tools: list[str] | None = None
    output_contract: ArtifactContract | None = None
    input_requirements: list[ArtifactRequirement] | None = None
    output_field_renames: list[OutputFieldRename] | None = None
    input_field_renames: list[InputFieldRename] | None = None


def _preserve_output_contract(
    previous: ArtifactContract | None,
    proposed: ArtifactContract | None,
    renames: list[dict] | None,
    *,
    contract_name: str = "output_contract",
    rename_field: str = "output_field_renames",
) -> None:
    """Preserve each declared obligation, allowing explicit non-merging renames.

    This checks structural preservation, not semantic equivalence of renamed
    fields or factual correctness of the resulting model output.
    """
    if previous is None:
        if renames:
            raise ValueError("Cannot rename fields without an existing output_contract")
        return
    if proposed is None:
        raise ValueError("Repair must preserve the existing output_contract")
    existing = set(previous.required_fields) | set(previous.identity_fields)
    mapping: dict[str, str] = {}
    for raw in renames or []:
        rename = OutputFieldRename.model_validate(raw)
        if rename.old_path not in existing or rename.old_path in mapping:
            raise ValueError(
                f"Unknown or duplicate output field rename: {rename.old_path!r}"
            )
        if not rename.new_path.strip():
            raise ValueError("Output field rename needs a non-empty new_path")
        mapping[rename.old_path] = rename.new_path
    paths = sorted(existing)
    targets = [mapping.get(path, path) for path in paths]
    if len(set(targets)) != len(targets):
        raise ValueError("Output field renames must not merge distinct obligations")

    def covers(container: str, field: str) -> bool:
        return field == container or field.startswith(
            (container + ".", container + "[].")
        )

    for index, left in enumerate(paths):
        for offset in range(index + 1, len(paths)):
            right = paths[offset]
            if not (covers(left, right) or covers(right, left)) and (
                covers(targets[index], targets[offset])
                or covers(targets[offset], targets[index])
            ):
                raise ValueError(
                    "Output field renames must not merge distinct obligations"
                )
    for kind in ("required_fields", "identity_fields"):
        declared = set(getattr(proposed, kind))
        for old_path in getattr(previous, kind):
            path = mapping.get(old_path, old_path)
            # A child guarantees its container; the converse would weaken a
            # requirement such as evidence.url to a merely non-empty evidence.
            if not any(covers(path, field) for field in declared):
                raise ValueError(
                    f"Repair must preserve {contract_name} {kind} field {old_path!r}. "
                    f"For a renamed field, supply {rename_field} and retain "
                    "the corresponding requirement in the new contract."
                )


def _preserve_input_requirements(
    previous: list[ArtifactRequirement],
    proposed: list[ArtifactRequirement],
    renames: list[dict] | None,
) -> None:
    """Preserve every obligation per source, even across split requirements."""

    def grouped(requirements: list[ArtifactRequirement]) -> dict[str, ArtifactContract]:
        result: dict[str, ArtifactContract] = {}
        for requirement in requirements:
            contract = result.setdefault(requirement.source_key, ArtifactContract())
            contract.required_fields.extend(requirement.required_fields)
            contract.identity_fields.extend(requirement.identity_fields)
        return result

    old = grouped(previous)
    new = grouped(proposed)
    by_source: dict[str, list[dict]] = {}
    for raw in renames or []:
        rename = InputFieldRename.model_validate(raw)
        if rename.source_key not in old:
            raise ValueError(f"Unknown input rename source: {rename.source_key!r}")
        by_source.setdefault(rename.source_key, []).append(
            rename.model_dump(exclude={"source_key"})
        )
    for source, contract in old.items():
        if source not in new:
            raise ValueError("Repair must preserve existing handoff dependencies")
        field_renames = by_source.get(source, [])
        try:
            _preserve_output_contract(
                contract,
                new[source],
                field_renames,
                contract_name="input_requirements",
                rename_field="input_field_renames",
            )
            mapping = {r["old_path"]: r["new_path"] for r in field_renames}
            # A child guarantees presence of its container, but equality of a
            # child does not preserve identity of the whole original object.
            for path in contract.identity_fields:
                if mapping.get(path, path) not in new[source].identity_fields:
                    raise ValueError(f"Repair must preserve identity field {path!r}")
        except ValueError as exc:
            raise ValueError(
                f"input_requirements for {source!r}: {exc}. "
                "Use input_field_renames with source_key/old_path/new_path "
                "for an explicit relocation."
            ) from exc


async def repair_agent_configs(
    tool_context: ToolContext, updates: list[AgentConfigRepair], reasoning: str
) -> str:
    """Atomically repair instructions, tools and contracts of related agents.

    Pass updates as an array of objects, NOT a JSON-encoded string. Omitted or
    null fields stay unchanged, including output_contract; null never deletes a
    contract. Empty lists are explicit edits, subject to preservation checks.
    Preserve state references such as {source?} in
    instructions. Include producer and consumer together for handoff changes.
    Names, output keys, models, pipeline and required dependencies cannot change.
    Existing output requirements cannot be removed. For a field rename supply
    output_field_renames=[{old_path: ..., new_path: ...}]; preserve every obligation
    in its original required_fields/identity_fields category. Do not merge fields.
    Preserve every input requirement too. Input path changes require
    input_field_renames=[{source_key: ..., old_path: ..., new_path: ...}]
    alongside input_requirements; update the producer if its shape changes.
    """
    try:
        payload = []
        for update in updates:
            item = AgentConfigRepair.model_validate(update).model_dump(
                exclude_unset=True
            )
            # The tool schema permits null placeholders for unchanged fields.
            # Normalize only the top-level patch, after validation: malformed
            # nested values and unknown fields must still be rejected.
            payload.append(
                {key: value for key, value in item.items() if value is not None}
            )
    except (ValueError, TypeError) as exc:
        return f"Error applying repair (configuration unchanged): {exc}"
    # Keep legacy JSON calls and all non-null edit guardrails unchanged.
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
    Existing output obligations must remain. Renamed paths require explicit
    output_field_renames=[{"old_path": ..., "new_path": ...}] in the same update.
    Input obligations also cannot be removed; input path changes require
    input_field_renames=[{"source_key": ..., "old_path": ..., "new_path": ...}].
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
        contract_updates: dict[str, list[dict] | None] = {}
        input_updates: dict[str, list[dict] | None] = {}
        for update in updates:
            if not isinstance(update, dict):
                raise TypeError("Every update must be an object")
            name = update.get("agent_name")
            if not isinstance(name, str) or name not in agents or name in touched:
                raise ValueError(f"Unknown or duplicate agent_name: {name!r}")
            fields = set(update) - {"agent_name"}
            if not fields or fields - (
                _EDITABLE_FIELDS | {"output_field_renames", "input_field_renames"}
            ):
                raise ValueError(
                    f"Only these fields may change: {sorted(_EDITABLE_FIELDS)}; "
                    "field renames may accompany their corresponding contract update"
                )
            if "output_field_renames" in fields and "output_contract" not in fields:
                raise ValueError(
                    "output_field_renames requires an output_contract update"
                )
            if "output_contract" in fields:
                renames = update.get("output_field_renames")
                if renames is not None and not isinstance(renames, list):
                    raise ValueError("output_field_renames must be an array of objects")
                contract_updates[name] = renames
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
                renames = update.get("input_field_renames")
                if renames is not None and not isinstance(renames, list):
                    raise ValueError("input_field_renames must be an array of objects")
                input_updates[name] = renames
            elif "input_field_renames" in fields:
                raise ValueError(
                    "input_field_renames requires an input_requirements update"
                )
            agents[name].update({key: update[key] for key in fields & _EDITABLE_FIELDS})
            touched.add(name)
        candidate = MAWConfig.model_validate(payload)
        old_agents = {agent.name: agent for agent in config.agents}
        for agent in candidate.agents:
            if agent.name in input_updates:
                _preserve_input_requirements(
                    old_agents[agent.name].input_requirements,
                    agent.input_requirements,
                    input_updates[agent.name],
                )
            if agent.name in contract_updates:
                _preserve_output_contract(
                    old_agents[agent.name].output_contract,
                    agent.output_contract,
                    contract_updates[agent.name],
                )
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
