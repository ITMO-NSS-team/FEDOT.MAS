"""Restrict a generation retry to missing, explicitly declared identity mappings."""

import copy
import json

from pydantic import ValidationError

from fedotmas.maw.models import MAWAgentConfig, MAWConfig, MAWStepConfig


def identity_repair_payload(error: Exception) -> dict | None:
    if not isinstance(error, ValidationError):
        return None
    errors = error.errors(include_url=False)
    if not errors or any("ambiguous identity paths" not in e["msg"] for e in errors):
        return None
    payload = errors[0].get("input")
    if not isinstance(payload, dict) or not isinstance(payload.get("agents"), list):
        return None
    return copy.deepcopy(payload)


def identity_repair_message(task: str, payload: dict, error: Exception) -> str:
    return (
        task
        + "\n\nTARGETED IDENTITY MAPPING REPAIR:\n"
        + str(error)
        + "\nReturn the same complete configuration, adding only the missing "
        "identity_field_mapping entries to the affected input_requirements. "
        "Each entry must have source_path and output_path. Preserve instructions, "
        "tools, evidence requirements, all required/identity fields, agent names, "
        "output keys and pipeline. A shared leaf name alone does not prove two "
        "identities are equivalent: use the intended data flow in the task and "
        "roles. Do not invent a mapping or delete an obligation to silence the "
        "validator. If the identity relation cannot be established, leave the "
        "configuration unresolved rather than guessing.\nCONFIGURATION:\n"
        + json.dumps(payload, ensure_ascii=False)
    )


def validate_identity_only_repair(original: dict, raw: dict | str) -> None:
    candidate = (
        MAWConfig.model_validate_json(raw)
        if isinstance(raw, str)
        else MAWConfig.model_validate(raw)
    )
    before_agents = [MAWAgentConfig.model_validate(a) for a in original["agents"]]
    if len(candidate.agents) != len(before_agents):
        raise ValueError(
            "Identity repair may only add missing identity_field_mapping entries"
        )
    cleaned = candidate.model_dump()
    for before, after in zip(before_agents, cleaned["agents"], strict=True):
        if len(before.input_requirements) != len(after["input_requirements"]):
            raise ValueError("Identity repair must preserve input requirements")
        for requirement, proposed in zip(
            before.input_requirements, after["input_requirements"], strict=True
        ):
            previous = [m.model_dump() for m in requirement.identity_field_mapping]
            added = [m for m in proposed["identity_field_mapping"] if m not in previous]
            for mapping in added:
                source, target = mapping["source_path"], mapping["output_path"]
                destinations = (
                    before.output_contract.identity_fields
                    if before.output_contract
                    else []
                )
                if (
                    source not in requirement.identity_fields
                    or source in destinations
                    or source.rsplit(".", 1)[-1] != target.rsplit(".", 1)[-1]
                    or source.count("[]") != target.count("[]")
                    or target not in destinations
                    or any(
                        m.source_path == source
                        for m in requirement.identity_field_mapping
                    )
                ):
                    raise ValueError(
                        "Identity repair changed an existing or unrelated mapping"
                    )
            proposed["identity_field_mapping"] = [
                m for m in proposed["identity_field_mapping"] if m not in added
            ]
    baseline = {
        "agents": [a.model_dump() for a in before_agents],
        "pipeline": MAWStepConfig.model_validate(original["pipeline"]).model_dump(),
        "final_answer_agent": original.get("final_answer_agent"),
    }
    if cleaned != baseline:
        raise ValueError(
            "Identity repair may only add missing identity_field_mapping entries; all other configuration must remain unchanged"
        )
