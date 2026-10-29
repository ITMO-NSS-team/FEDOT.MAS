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
Plan a retrieval fallback using the tools actually available: browser navigation
is not downloading a CSV/ZIP/API response. Use download for file resources, then
document tools to list/extract archives and read the returned local paths. Never
treat a successful navigation message as downloaded data. For a blocked page,
try an authoritative alternative or a different reading backend once; do not
keep changing selectors on an access-denied/error page. An empty extraction
does not prove that the requested fact is absent. Record failed sources so a
downstream role can select an alternative without repeating the same attempts.
For unfamiliar symbols or encodings, assign a computation tool to inspect their
codepoints and a lookup tool to verify meanings; do not ask the user to supply
information already present in the input. Tool-free roles must have sufficient
evidence to complete their assigned task.
When evidence is missing, use status=unresolved with a reason in an intermediate
artifact, or the terminal abstention protocol. Null result fields plus prose are
not a completed answer. Never use benchmark answer dumps as evidence for a fact.
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
