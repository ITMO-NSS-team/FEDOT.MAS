"""Keep declared source obligations stable across generation and recovery."""

import json
import re

from fedotmas.maw.models import MAWAgentConfig, MAWConfig
from fedotmas.maw.contracts import field_path_is_guaranteed
from fedotmas.maw._validators import _find_terminal_node


EVIDENCE_PLANNING_GUIDANCE = """
SOURCE OBLIGATIONS:
For each source the task requires you to read (a named publication, URL, supplied
file or archive), put evidence_requirements on the agent that actually reads it.
Each item has task_quote (verbatim task excerpt), source (verbatim source reference
within that excerpt), source_kind (url, file, or reference for a named source),
source_field (the output field containing the actually read URL or file path),
and required_fields (the facts/formula/units/bindings needed from that source).
Do not invent a URL or filename before inspecting the supplied archive or source.
Use source_kind=reference for descriptions such as 'the attached spreadsheet'.
When the task supplies a Host file path, retain that exact path with source_kind=file;
reading an extracted member of that archive can fulfill the archive obligation.
Include source_field and required_fields in output_contract.required_fields.
The source reader must have inspection tools and a matching output_contract;
search-only and download-only roles cannot fulfill source obligations. Wire its
required fields into downstream input_requirements before computing an answer.
Keep the final presentation stage separate from these structured extractions.
Inventory archives before assigning contained files: a job description PDF and
an applicant spreadsheet have different roles. Inspect XML structure before
choosing tags; a visible label need not be an XML element name.
If the task specifies an equation from a publication, retrieve the equation,
variable definitions and units before selecting a calculation. Preserve input
column names and their documented meanings; a rate constant is not automatically
a maximum rate. Missing units or conversion factors remain unresolved.
Recovery may change retrieval routes and fix schema wiring; it cannot delete
source obligations or turn assumed values into observed facts. Return unresolved
when a required source or field cannot be established.
""".strip()


def evidence_instruction(cfg: MAWAgentConfig) -> str | None:
    if not cfg.evidence_requirements:
        return None
    return (
        "REQUIRED SOURCE EVIDENCE (retained independently of role instructions):\n"
        + json.dumps(
            [r.model_dump() for r in cfg.evidence_requirements], ensure_ascii=False
        )
        + "\nRead each required source with an actual tool. Do not replace missing "
        "facts, formulas, variable bindings or units with assumptions. Report "
        "unresolved when the evidence cannot be obtained."
    )


def validate_task_evidence(config: MAWConfig, task: str) -> None:
    # Explicit attachment metadata is mechanically checkable. Do not infer all
    # natural-language source obligations with a keyword heuristic.
    files = re.findall(r"(?m)^Host file path:\s*(/[^\n]+)$", task)
    declared_files = {
        r.source
        for a in config.agents
        for r in a.evidence_requirements
        if r.source_kind == "file"
    }
    missing = set(files) - declared_files
    if missing:
        raise ValueError(
            f"No evidence requirement for supplied host files: {sorted(missing)}"
        )
    for agent in config.agents:
        for requirement in agent.evidence_requirements:
            if requirement.task_quote not in task:
                raise ValueError(
                    f"Agent '{agent.name}' evidence task_quote is not a verbatim excerpt of the original task"
                )


def preserve_task_evidence(old: MAWConfig, new: MAWConfig) -> None:
    agents = {a.name: a for a in new.agents}
    for previous in old.agents:
        if not previous.evidence_requirements:
            continue
        replacement = agents.get(previous.name)
        if replacement is None or any(
            r not in replacement.evidence_requirements
            for r in previous.evidence_requirements
        ):
            raise ValueError(
                f"Recovery must preserve source obligations for '{previous.name}'"
            )


def evidence_flow_errors(config: MAWConfig) -> list[str]:
    producers = {a.output_key: a for a in config.agents if a.evidence_requirements}
    if not producers:
        return []
    errors = []
    agents = {a.name: a for a in config.agents}
    seen = set()
    consumed: dict[str, set[str]] = {}

    def walk(node, available):
        if node.type == "agent":
            agent = agents[node.agent_name]
            seen.add(agent.name)
            for requirement in agent.input_requirements:
                if requirement.source_key in producers:
                    if requirement.source_key not in available:
                        errors.append(
                            f"Evidence '{requirement.source_key}' must be produced before '{agent.name}', not in a parallel or later branch"
                        )
                    consumed.setdefault(requirement.source_key, set()).update(
                        [
                            *requirement.required_fields,
                            *requirement.identity_fields,
                        ]
                    )
            return available | {agent.output_key}
        if node.type == "parallel":
            return set(available).union(
                *(walk(child, set(available)) for child in node.children)
            )
        for child in node.children:
            available = walk(child, available)
        return available

    walk(config.pipeline, set())
    terminal = _find_terminal_node(config.pipeline).agent_name
    for key, producer in producers.items():
        if producer.name not in seen:
            errors.append(
                f"Evidence producer '{producer.name}' is not executed by the pipeline"
            )
        if producer.name == terminal:
            continue
        needed = {
            field
            for r in producer.evidence_requirements
            for field in [*r.required_fields, r.source_field]
        }
        missing = {
            field
            for field in needed
            if not any(
                field_path_is_guaranteed(field, path)
                for path in consumed.get(key, set())
            )
        }
        if missing:
            errors.append(
                f"Evidence from '{producer.name}' must be consumed through input_requirements, including {sorted(missing)}"
            )
    return errors
