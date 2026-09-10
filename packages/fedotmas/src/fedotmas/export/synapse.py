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
from hashlib import blake2s
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

#: What their import writes onto an existing agent whatever the bundle says:
#: fields their schema requires, fields only this side knows, and the model
#: trio ``materialize_agent_model_params`` fills in before the upsert.
_OVERWRITTEN_ON_REUSE = (
    "type",
    "system_prompt",
    "model",
    "temperature",
    "reasoning_effort",
    "allowed_tools",
    "allowed_mcp_tools",
    "output_save_key",
)

#: Room kept for the workflow namespace inside their 64-character identifier.
#: Whatever is left goes to the agent name, which may be shortened; the
#: namespace may not, since it is what keeps one workflow out of another's
#: records.
_SCOPE_BUDGET = 40

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
    #: Fields of a reused agent their import overwrites even though the bundle
    #: leaves them out.  Empty when nothing is reused.
    overwritten_fields: tuple[str, ...] = ()
    #: Every node reads the whole state.  Ours flows through named keys quoted in
    #: instructions, so an exported agent sees more than it was wired to.  True
    #: on every bundle this module emits today.
    wildcard_reads: bool = True


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


def _workflow_scope(wire_workflow_id: str) -> str:
    """Return the namespace part of a generated agent's wire name.

    A workflow id too long to carry whole is cut and fingerprinted rather than
    simply truncated: two ids that differ only past the cut would otherwise
    share a namespace, which is the collision this exists to prevent.
    """
    if len(wire_workflow_id) <= _SCOPE_BUDGET:
        return wire_workflow_id
    digest = blake2s(wire_workflow_id.encode(), digest_size=4).hexdigest()
    return f"{wire_workflow_id[: _SCOPE_BUDGET - 9].rstrip('_')}_{digest}"


def _scoped_wire_name(scope: str, name: str, taken: set[str]) -> str:
    """Return a wire name for *name*, namespaced under the workflow *scope*.

    Agent identity is tenant-wide over there while ours is per-config, so a
    second bundle carrying the obvious ``researcher`` would import as an edit of
    the first one's agent.  Their own bundles are namespaced by hand
    (``urban_planner``, ``urban_zoning_finder``); this does the same by machine.
    A name with no room left is shortened — ``taken`` still keeps it unique
    inside the bundle, and the namespace it sits under is intact.
    """
    slug = to_wire_name(name)[: 63 - len(scope)].rstrip("_")
    return to_wire_name(f"{scope}_{slug}" if slug else scope, taken)


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
            record rather than creating a second one, and that agent is emitted
            with only the fields this side owns — see ``overwritten_fields``.
        tool_catalog: The catalogue generation ran against. Tools outside it are
            dropped and reported: an unknown id imports as a broken reference,
            leaving the agent with a silently empty tool list at run time.
        phase_label: Their phase indicator. The requirements/planning/execution/
            output quartet is one bundle's convention, not a fixed vocabulary.
    """
    external = {a.name: a for a in existing_agents.agents} if existing_agents else {}
    wire_workflow_id = to_wire_name(workflow_id)
    scope = _workflow_scope(wire_workflow_id)

    taken: set[str] = set()
    wire: dict[str, str] = {}
    reused: list[str] = []
    reused_names: set[str] = set()
    renamed: list[tuple[str, str]] = []
    for agent in config.agents:
        entry = external.get(agent.name)
        given = entry.id if entry is not None and entry.id else None
        wire[agent.name] = (
            to_wire_name(given, taken)
            if given is not None
            else _scoped_wire_name(scope, agent.name, taken)
        )
        if given is None:
            continue
        # Only an id that survived untouched still points at their record; a
        # slugified one silently becomes a second one beside it.
        if wire[agent.name] == given:
            reused.append(given)
            reused_names.add(agent.name)
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
            reused=agent.name in reused_names,
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

    workflow = {
        "_id": wire_workflow_id,
        "name": workflow_name or wire_workflow_id,
        "description": f"Generated by FEDOT.MAS from {len(config.agents)} agents",
        # Their per-node `reads` would have to be mined out of our prompts, where
        # data flow lives as {key} references; "*" is what their own bundle uses,
        # and `wildcard_reads` reports what it costs.
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
        overwritten_fields=_OVERWRITTEN_ON_REUSE if reused else (),
    )


def _agent_doc(
    agent: MAWAgentConfig,
    wire_id: str,
    *,
    reused: bool,
    temperature: float,
    phase_label: str,
    tool_catalog: dict[str, str] | None,
    unresolved: list[str],
) -> dict[str, Any]:
    """Build the agent record their import writes.

    Their import is a field-wise ``$set`` keyed on the agent's name, so every
    field emitted here lands on whatever record already answers to it.  A reused
    agent is the caller's own, and gets only what their schema requires plus what
    this side alone knows; the rest of their record survives the import.  An
    empty ``allowed_tools`` is not among the fields that may be left out — their
    import normalizes the pair of tool lists out of whatever arrives, so an
    absent one would clear the tools the agent already has.  A reused agent's
    tools go out unfiltered for the same reason: they were assigned on their
    side, and a catalogue narrower than their tenant would strip a live agent of
    tools it is using.
    """
    if tool_catalog is None or reused:
        tools = list(agent.tools)
        if reused and tool_catalog is not None:
            outside = [t for t in tools if t not in tool_catalog]
            if outside:
                _log.warning(
                    "Reused agent {!r} keeps tools the catalogue does not list: {}",
                    wire_id,
                    outside,
                )
    else:
        tools = [t for t in agent.tools if t in tool_catalog]
        unresolved.extend(t for t in agent.tools if t not in tool_catalog)

    doc: dict[str, Any] = {
        "_id": wire_id,
        # `name` is the identity their import matches on and `type` an auction
        # kind, which nothing we emit uses (configuration_schemas.py:246-250).
        "name": wire_id,
        "type": wire_id,
        "model": agent.model,
        "allowed_tools": tools,
        "output_save_key": agent.output_key,
        "system_prompt": _plain_prompt(agent.instruction),
    }
    if reused:
        return doc
    doc |= {
        "description": _describe(agent.instruction),
        "agent_class": "GenericAgent",
        "temperature": temperature,
        "allowed_phases": [phase_label],
        "allowed_delegation_targets": None,
        "enabled": True,
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
        # Node ids live inside the workflow, so they keep the plain agent name;
        # only `agent_type` needs the tenant-wide identity.
        node_id = to_wire_name(agent.name, walk.used)
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
