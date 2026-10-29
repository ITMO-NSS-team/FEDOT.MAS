"""Bound a generation repair to source declarations and additive field wiring."""

from copy import deepcopy
import json

from fedotmas.maw.models import MAWAgentConfig, MAWConfig, MAWStepConfig
from fedotmas.maw._evidence_requirements import validate_task_evidence


def evidence_repair_payload(raw, error: Exception) -> dict | None:
    if (
        "evidence" not in str(error).lower()
        and "source obligations" not in str(error).lower()
    ):
        return None
    try:
        payload = json.loads(raw) if isinstance(raw, str) else deepcopy(raw)
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("agents"), list):
        return None
    return payload


def evidence_repair_message(task: str, payload: dict, error: Exception) -> str:
    return (
        task
        + "\n\nTARGETED SOURCE DECLARATION REPAIR:\n"
        + str(error)
        + "\nReturn the same complete configuration. Preserve agent names, roles, "
        "instructions, tools, models, pipeline, identities, calculation requirements "
        "and ALL required facts. Do not remove evidence_requirements. Correct only "
        "invalid task_quote/source/source_kind/source_field and add missing field "
        "paths to producer output_contract and consumer input_requirements. "
        "task_quote is a literal contiguous excerpt of the ORIGINAL TASK; source "
        "is a literal substring of it. A named source uses reference, not url. "
        "source_field is a path inside the READER'S output (e.g. source_url), "
        "never user_query or finder_result.video_url. Copy the actually read URL "
        "into that reader field. Require that path and every evidence fact in "
        "both the reader output and downstream consumer input. Preserve already "
        "valid task references. Do not guess source contents or weaken checks. "
        "If this limited repair cannot work, leave the conflict explicit.\n"
        "CONFIGURATION TO REPAIR:\n" + json.dumps(payload, ensure_ascii=False)
    )


def validate_evidence_only_repair(original: dict, raw, task: str) -> None:
    proposed = json.loads(raw) if isinstance(raw, str) else deepcopy(raw)
    candidate = MAWConfig.model_validate(deepcopy(proposed))
    validate_task_evidence(candidate, task)
    if (
        len(original["agents"]) != len(candidate.agents)
        or MAWStepConfig.model_validate(original["pipeline"]) != candidate.pipeline
        or original.get("final_answer_agent") != candidate.final_answer_agent
    ):
        raise ValueError("Source repair must preserve agents and pipeline")
    producer_keys = {a.output_key for a in candidate.agents if a.evidence_requirements}
    baseline = deepcopy(original)
    for agent in baseline["agents"]:
        agent["evidence_requirements"] = []
    # Apply the same existing defaults and inferred passthrough IDs to both.
    baseline = MAWConfig.model_validate(baseline)
    for before, normalized, after in zip(
        original["agents"], baseline.agents, candidate.agents, strict=True
    ):
        previous = before.get("evidence_requirements") or []
        if len(previous) > len(after.evidence_requirements):
            raise ValueError("Source repair must preserve every source obligation")
        for new in after.evidence_requirements:
            if new.source_field == "user_query" or any(
                new.source_field.startswith(a.output_key + ".")
                for a in candidate.agents
                if a.name != after.name
            ):
                raise ValueError("Source repair must use the reader's own output field")
        for old, new in zip(previous, after.evidence_requirements):
            declared = (before.get("output_contract") or {}).get("required_fields", [])
            old_locator = old["source_field"]
            if (
                old_locator in declared
                and old_locator != "user_query"
                and not any(
                    old_locator.startswith(a.output_key + ".")
                    for a in candidate.agents
                    if a.name != after.name
                )
                and old_locator != new.source_field
            ):
                raise ValueError("Source repair must preserve valid locator fields")
            if old["required_fields"] != new.required_fields:
                raise ValueError("Source repair must preserve required evidence facts")
            if old["task_quote"] in task and old["task_quote"] != new.task_quote:
                raise ValueError("Source repair must preserve valid task quotes")
            if old["source"] in task and old["source"] != new.source:
                raise ValueError("Source repair must preserve valid source references")
            # A valid literal source type cannot be downgraded to a reference.
            if (
                old["source"] in task
                and (
                    old["source_kind"] == "file"
                    or old["source_kind"] == "reference"
                    or old["source"].startswith(("http://", "https://"))
                )
                and old["source_kind"] != new.source_kind
            ):
                raise ValueError("Source repair must preserve valid source kinds")
        left, right = normalized.model_dump(), after.model_dump()
        left["evidence_requirements"] = right["evidence_requirements"] = []
        old_contract = left.get("output_contract")
        new_contract = right.get("output_contract")
        if old_contract:
            if not new_contract or not set(
                old_contract.get("required_fields", [])
            ) <= set(new_contract["required_fields"]):
                raise ValueError("Source repair cannot remove output fields")
            new_contract["required_fields"] = old_contract.get("required_fields", [])
        elif new_contract:
            if not after.evidence_requirements or new_contract.get("identity_fields"):
                raise ValueError("Source repair cannot introduce unrelated contracts")
            right["output_contract"] = None
        old_inputs = left.get("input_requirements") or []
        new_inputs = right["input_requirements"]
        if len(new_inputs) < len(old_inputs):
            raise ValueError("Source repair cannot remove input requirements")
        for old, new in zip(old_inputs, new_inputs):
            if not set(old.get("required_fields", [])) <= set(new["required_fields"]):
                raise ValueError("Source repair cannot remove input fields")
            new["required_fields"] = old.get("required_fields", [])
        for added in new_inputs[len(old_inputs) :]:
            if (
                added["source_key"] not in producer_keys
                or added["identity_fields"]
                or added["identity_field_mapping"]
            ):
                raise ValueError("Source repair can only add source evidence inputs")
        right["input_requirements"] = new_inputs[: len(old_inputs)]
        if MAWAgentConfig.model_validate(left) != MAWAgentConfig.model_validate(right):
            raise ValueError(
                "Source repair may only correct declarations and add required fields"
            )
