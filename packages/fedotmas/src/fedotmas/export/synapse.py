"""Emit a ``MAWConfig`` as a Synapse configuration bundle.

The target is the JSON that Synapse's ``/api/admin/config-bundle`` imports:
``{version, exported_from, items}`` with ``items`` keyed by kind.  Their engine
executes it; nothing here builds or runs anything.

``parallel`` and ``loop`` have no counterpart there and are converted rather
than copied; ``SynapseExport`` reports what that cost.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from fedotmas._settings import get_max_loop_iterations
from fedotmas.common.logging import get_logger
from fedotmas.maw.models import (
    AgentPoolConfig,
    MAWAgentConfig,
    MAWConfig,
    MAWStepConfig,
)

_log = get_logger("fedotmas.export.synapse")

#: Their bundle format version (``bundle_service.BUNDLE_VERSION``).
BUNDLE_VERSION = 1

#: Identity of every configuration item they import: ``^[a-z][a-z0-9_]{1,63}$``.
_WIRE_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{1,63}$")

#: Their per-validator reject cap is bounded at save time.
_MAX_REJECT_RETRIES = 10

#: Our own optional-state syntax; meaningless outside this runtime.
_OPTIONAL_STATE_REF_RE = re.compile(r"\{(\w+)\?\}")

#: Their identifiers are ASCII, and this integration's agents are named in
#: Russian; without this every such name slugs to nothing.
_TRANSLIT = str.maketrans(
    {
        "а": "a",
        "б": "b",
        "в": "v",
        "г": "g",
        "д": "d",
        "е": "e",
        "ё": "e",
        "ж": "zh",
        "з": "z",
        "и": "i",
        "й": "i",
        "к": "k",
        "л": "l",
        "м": "m",
        "н": "n",
        "о": "o",
        "п": "p",
        "р": "r",
        "с": "s",
        "т": "t",
        "у": "u",
        "ф": "f",
        "х": "kh",
        "ц": "ts",
        "ч": "ch",
        "ш": "sh",
        "щ": "shch",
        "ъ": "",
        "ы": "y",
        "ь": "",
        "э": "e",
        "ю": "yu",
        "я": "ya",
    }
)


@dataclass
class SynapseExport:
    """A bundle plus what converting to it cost."""

    bundle: dict[str, Any]
    #: Branches pulled out of ``parallel`` blocks into the chain.
    linearized_branches: int = 0
    #: Tools dropped because the caller's catalogue does not list them.
    unresolved_tools: tuple[str, ...] = ()
    #: Caller ids kept verbatim, so their import updates those records.
    reused_agents: tuple[str, ...] = ()
    #: Caller ids their format would reject, as ``(given, emitted)``.  Each will
    #: import as a new record rather than an update.
    renamed_ids: tuple[tuple[str, str], ...] = ()
    #: Loops whose gate cannot reject, so they run once.  Their validator judges
    #: structure, not content, and any non-empty result passes it.
    degraded_loops: int = 0
    #: Supplied agents the config does not name, so their ids went nowhere.  Work
    #: they were meant to do is now carried by a freshly minted agent beside the
    #: record the platform already has.
    unmatched_agents: tuple[str, ...] = ()


@dataclass
class _Walk:
    """Accumulator for one pass over the pipeline tree."""

    nodes: list[dict[str, Any]] = field(default_factory=list)
    edges: list[dict[str, Any]] = field(default_factory=list)
    ids: dict[str, MAWAgentConfig] = field(default_factory=dict)
    #: Node ids already taken, the two sentinels included.
    used: set[str] = field(default_factory=lambda: {"start", "end"})
    #: Nodes whose outgoing forward edge must say ``approved`` — validators.
    branching: set[str] = field(default_factory=set)
    linearized: int = 0
    degraded_loops: int = 0


def to_wire_name(value: str, taken: set[str] | None = None) -> str:
    """Return *value* as an identifier Synapse accepts, unique within *taken*."""
    slug = value.strip()
    if not _WIRE_NAME_RE.match(slug):
        slug = re.sub(r"[^a-z0-9]+", "_", slug.lower().translate(_TRANSLIT)).strip("_")
    if not slug or not slug[0].isalpha():
        slug = f"a_{slug}" if slug else "agent"
    slug = slug[:64]
    if len(slug) < 2:
        slug = f"{slug}_1"

    if taken is None:
        return slug
    candidate, n = slug, 2
    while candidate in taken:
        suffix = f"_{n}"
        candidate = f"{slug[: 64 - len(suffix)]}{suffix}"
        n += 1
    taken.add(candidate)
    return candidate


def to_synapse_bundle(
    config: MAWConfig,
    *,
    workflow_id: str,
    workflow_name: str | None = None,
    existing_agents: AgentPoolConfig | None = None,
    tool_catalog: dict[str, str] | None = None,
    temperature: float = 1.0,
    phase_label: str = "execution",
) -> SynapseExport:
    """Convert *config* into a bundle their config-import accepts.

    Args:
        workflow_id: Wire id for the workflow; also its name on import.
        existing_agents: The pool this config was generated over, if any. An
            entry's ``id`` is reused verbatim so the import updates their
            record rather than creating a second one.
        tool_catalog: The catalogue generation ran against. Tools outside it are
            dropped and reported: an unknown id imports as a broken reference,
            leaving the agent with a silently empty tool list at run time.
        phase_label: Their phase indicator. The requirements/planning/execution/
            output quartet is one bundle's convention, not a fixed vocabulary.
    """
    external = {a.name: a for a in existing_agents.agents} if existing_agents else {}

    taken: set[str] = set()
    wire: dict[str, str] = {}
    reused: list[str] = []
    renamed: list[tuple[str, str]] = []
    for agent in config.agents:
        entry = external.get(agent.name)
        given = entry.id if entry is not None and entry.id else None
        wire[agent.name] = to_wire_name(given or agent.name, taken)
        if given is None:
            continue
        # Only an id that survived untouched still points at their record; a
        # slugified one silently becomes a second one beside it.
        if wire[agent.name] == given:
            reused.append(given)
        else:
            renamed.append((given, wire[agent.name]))
            _log.warning(
                "Agent id {!r} is not a name Synapse accepts; emitted as {!r}, "
                "which will import as a new record",
                given,
                wire[agent.name],
            )

    unmatched = tuple(sorted(external.keys() - {a.name for a in config.agents}))
    if unmatched:
        _log.warning(
            "Supplied agents missing from the config: {}; their ids cannot be "
            "carried and the work goes to newly created records",
            unmatched,
        )

    unresolved: list[str] = []
    agents = [
        _agent_doc(
            agent,
            wire[agent.name],
            temperature=temperature,
            phase_label=phase_label,
            tool_catalog=tool_catalog,
            unresolved=unresolved,
        )
        for agent in config.agents
    ]

    walk = _Walk(ids={a.name: a for a in config.agents})
    first, last = _emit(config.pipeline, walk, wire, phase_label)
    walk.nodes.insert(0, {"id": "start", "type": "start"})
    walk.nodes.append({"id": "end", "type": "end"})
    walk.edges.insert(0, {"from": "start", "to": first})
    walk.edges.append(_forward_edge(last, "end", walk))
    order = {node["id"]: i for i, node in enumerate(walk.nodes)}
    walk.edges.sort(key=lambda e: (order[e["from"]], order[e["to"]]))

    wire_workflow_id = to_wire_name(workflow_id)
    workflow = {
        "_id": wire_workflow_id,
        "name": workflow_name or wire_workflow_id,
        "description": f"Generated by FEDOT.MAS from {len(config.agents)} agents",
        # Their per-node `reads` would have to be mined out of our prompts, where
        # data flow lives as {key} references; "*" is what their own bundle uses.
        "default_reads": ["*"],
        "nodes": walk.nodes,
        "edges": walk.edges,
        "is_default": False,
    }

    _log.info(
        "Bundle emitted | agents={} nodes={} linearized={} degraded_loops={} "
        "unresolved_tools={}",
        len(agents),
        len(walk.nodes),
        walk.linearized,
        walk.degraded_loops,
        len(unresolved),
    )
    return SynapseExport(
        bundle={
            "version": BUNDLE_VERSION,
            "exported_from": "fedotmas",
            "items": {
                "agents": agents,
                "workflows": [workflow],
                "tools": [],
                "run_configurations": [],
            },
        },
        linearized_branches=walk.linearized,
        degraded_loops=walk.degraded_loops,
        unresolved_tools=tuple(dict.fromkeys(unresolved)),
        reused_agents=tuple(reused),
        renamed_ids=tuple(renamed),
        unmatched_agents=unmatched,
    )


def _agent_doc(
    agent: MAWAgentConfig,
    wire_id: str,
    *,
    temperature: float,
    phase_label: str,
    tool_catalog: dict[str, str] | None,
    unresolved: list[str],
) -> dict[str, Any]:
    if tool_catalog is None:
        tools = list(agent.tools)
    else:
        tools = [t for t in agent.tools if t in tool_catalog]
        unresolved.extend(t for t in agent.tools if t not in tool_catalog)

    doc: dict[str, Any] = {
        "_id": wire_id,
        # `name` is the identity their import matches on and `type` an auction
        # kind, which nothing we emit uses (configuration_schemas.py:246-250).
        "name": wire_id,
        "type": wire_id,
        "description": _describe(agent.instruction),
        "agent_class": "GenericAgent",
        "model": agent.model,
        "temperature": temperature,
        "allowed_phases": [phase_label],
        # Only an auction reads these, and every node we emit names its agent.
        "eval_keywords": [],
        "allowed_tools": tools,
        "allowed_delegation_targets": None,
        "output_save_key": agent.output_key,
        "enabled": True,
        "system_prompt": _plain_prompt(agent.instruction),
    }
    if agent.name != wire_id:
        doc["display_name"] = agent.name
    return doc


def _describe(instruction: str, limit: int = 200) -> str:
    """One line about the role, which their UI shows and our config lacks."""
    text = " ".join(_plain_prompt(instruction).split())
    return text[: limit - 1] + "…" if len(text) > limit else text


def _plain_prompt(instruction: str) -> str:
    """Drop our optional-state marker, which means nothing outside this runtime."""
    return _OPTIONAL_STATE_REF_RE.sub(r"{\1}", instruction)


def _emit(
    step: MAWStepConfig,
    walk: _Walk,
    wire: dict[str, str],
    phase_label: str,
) -> tuple[str, str]:
    """Append the nodes for *step* and return its (entry, exit) node ids."""
    if step.type == "agent":
        agent = walk.ids[str(step.agent_name)]
        node_id = to_wire_name(wire[agent.name], walk.used)
        node: dict[str, Any] = {
            "id": node_id,
            "type": "phase",
            "description": _describe(agent.instruction),
            "agent_selection": "direct",
            "agent_type": wire[agent.name],
            "phase_label": phase_label,
        }
        if agent.output_key:
            node["writes"] = [agent.output_key]
        walk.nodes.append(node)
        return node_id, node_id

    if step.type == "parallel":
        # One successor per node in their engine: the branches become a chain.
        walk.linearized += max(len(step.children) - 1, 0)

    first, last = _chain(step.children, walk, wire, phase_label)
    if step.type != "loop":
        return first, last

    last_agent = _last_agent(step, walk)
    check_key = last_agent.output_key if last_agent else None
    validator_id = to_wire_name(f"{last}_check", walk.used)
    validator: dict[str, Any] = {
        "id": validator_id,
        "type": "validator",
        "checks": [
            {
                "key": check_key or "result",
                "kind": "structural",
                "type": "string",
                "min_length": 1,
            }
        ],
    }
    # Ours counts passes and falls back to a configured default, theirs counts
    # rejections after the first and is capped at save time.
    iterations = step.max_iterations or get_max_loop_iterations()
    validator["max_reject_retries"] = max(0, min(iterations - 1, _MAX_REJECT_RETRIES))
    walk.nodes.append(validator)
    walk.degraded_loops += 1
    walk.branching.add(validator_id)
    walk.edges.append(_forward_edge(last, validator_id, walk))
    walk.edges.append({"from": validator_id, "to": first, "condition": "rejected"})
    return first, validator_id


def _chain(
    children: list[MAWStepConfig],
    walk: _Walk,
    wire: dict[str, str],
    phase_label: str,
) -> tuple[str, str]:
    first = last = ""
    for child in children:
        entry, exit_ = _emit(child, walk, wire, phase_label)
        if not first:
            first = entry
        else:
            walk.edges.append(_forward_edge(last, entry, walk))
        last = exit_
    return first, last


def _forward_edge(src: str, dst: str, walk: _Walk) -> dict[str, Any]:
    edge = {"from": src, "to": dst}
    if src in walk.branching:
        edge["condition"] = "approved"
    return edge


def _last_agent(step: MAWStepConfig, walk: _Walk) -> MAWAgentConfig | None:
    """Return the agent that finishes *step*, whose output a loop gates on."""
    if step.type == "agent":
        return walk.ids.get(str(step.agent_name))
    for child in reversed(step.children):
        found = _last_agent(child, walk)
        if found is not None:
            return found
    return None
