"""Keep the caller's task separate from generated plans and evidence artifacts."""

from __future__ import annotations


TASK_REQUIREMENTS_GUIDANCE = """
TASK FIDELITY:
The original user task defines the requested result. A generated role, plan,
handoff schema, or upstream answer must not redefine it. Preserve the requested
entity type, components, units, scale, rounding, ordering and scope. Keep useful
supporting details in internal evidence; do not add them to the caller-facing
answer unless requested. Distinguish base-unit intermediate values from values
in the requested scale, label both explicitly, and apply rounding only after
the required conversion. A request for thousands is not a request for units.
For an exact factual question, an unsupported estimate is missing evidence,
not an admissible input to calculation. Keep it unresolved until supported.
Before selecting a source, check the task's identifying constraints (such as
year, title, author, event or label). Search rank and a similar topic do not
establish identity. When a constraint cannot be checked, record the uncertainty
instead of treating a candidate as a verified source. Preserve checked source
identity through extraction; do not silently substitute another document.
Assign tools that can obtain the actual evidence modality. Reading HTML around
an image or a caption is not inspecting the image itself. If image evidence is
needed, an executed agent must have a working image-understanding tool; missing
capabilities are unresolved dependencies, not permission to guess.
""".strip()


def task_requirements_instruction(query: object, *, terminal: bool) -> str | None:
    """Append verbatim task text at request time, never interpret its braces."""
    if not isinstance(query, str) or not query.strip():
        return None
    boundary = (
        "This is the terminal answer stage. Determine the requested answer from "
        "the original task below; extra deliverables in generated instructions "
        "above are internal working notes, not additional requested outputs. "
        "Honor the caller's final answer format."
        if terminal
        else "Complete only your assigned subtask and its handoff contract. Keep "
        "the original task's meaning and requested scale intact for downstream "
        "agents; its final submission format belongs only at the terminal stage."
    )
    return (
        f"{TASK_REQUIREMENTS_GUIDANCE}\n{boundary}\n"
        f"ORIGINAL USER TASK (verbatim):\n{query}"
    )
