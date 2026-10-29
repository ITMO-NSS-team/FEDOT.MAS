from __future__ import annotations

import re
from copy import deepcopy
from enum import Enum
from typing import Any

from fedotmas.maw.handoffs import (
    ABSTENTION_STATE_KEY,
    describe_requirement,
    is_explicit_abstention,
    is_unresolved_answer,
    unresolved_execution_issues,
    validate_output_contract,
)
from fedotmas.maw.models import MAWConfig
from fedotmas.plugins._checkpoint import Checkpoint


class Strategy(str, Enum):
    RETRY_FAILED = "retry_failed"
    RESTART_AFTER = "restart_after"
    RESTART_ALL = "restart_all"


def checkpoint_usable(cp: Checkpoint, config: MAWConfig) -> bool:
    agent = next((a for a in config.agents if a.name == cp.agent_name), None)
    if agent is None or not cp.validated:
        return False
    value = cp.state.get(agent.output_key)
    if (
        value is None
        or value == ""
        or (isinstance(value, (dict, list)) and not value)
        or is_unresolved_answer(value)
        or is_explicit_abstention(value)
    ):
        return False
    if agent.output_contract and validate_output_contract(value, agent.output_contract):
        return False
    if any(describe_requirement(cp.state, r)[1] for r in agent.input_requirements):
        return False
    if any(
        i.get("agent") == cp.agent_name for i in unresolved_execution_issues(cp.state)
    ):
        return False
    if cp.agent_name in (cp.state.get("_fedotmas_execution") or {}).get(
        "limited_agents", {}
    ):
        return False
    completion = cp.state.get(ABSTENTION_STATE_KEY) or {}
    return not (
        completion.get("agent") == cp.agent_name
        and completion.get("status") in {"abstained", "unresolved"}
    )


def resolve_initial_state(
    strategy: Strategy,
    checkpoints: list[Checkpoint],
    old_config: object,
    new_config: object,
) -> tuple[dict[str, Any] | None, set[str]]:
    """Reuse only checked, unchanged outputs whose dependencies are also reusable.

    Preserve independent parallel branches. A changed workflow or a loop is
    conservatively rerun; no arbitrary last-session state is carried forward.
    """
    if not checkpoints or strategy == Strategy.RESTART_ALL:
        return None, set()
    if not isinstance(old_config, MAWConfig) or not isinstance(new_config, MAWConfig):
        return None, set()
    if old_config.pipeline != new_config.pipeline:
        return None, set()
    old = {a.name: a for a in old_config.agents}
    new = {a.name: a for a in new_config.agents}
    owners = {a.output_key: a.name for a in new_config.agents}
    dependencies: dict[str, set[str]] = {}
    loop_agents: set[str] = set()

    def walk(node, preceding, in_loop=False):
        if node.type == "agent":
            name = node.agent_name
            agent = new[name]
            refs = [r.source_key for r in agent.input_requirements]
            refs += re.findall(
                r"(?<!\{)\{(\w+)(?:\.\w+)*\??\}(?!\})", agent.instruction
            )
            dependencies.setdefault(name, set()).update(
                preceding
                | {owners[k] for k in refs if k in owners and owners[k] != name}
            )
            if in_loop:
                loop_agents.add(name)
            return preceding | {name}
        if node.type == "parallel":
            return set(preceding).union(
                *(walk(c, set(preceding), in_loop) for c in node.children)
            )
        for child in node.children:
            preceding = walk(child, preceding, in_loop or node.type == "loop")
        return preceding

    walk(new_config.pipeline, set())
    latest = {cp.agent_name: cp for cp in checkpoints}
    reusable = {
        name
        for name, cp in latest.items()
        if name in dependencies
        and name not in loop_agents
        and old.get(name) == new.get(name)
        and checkpoint_usable(cp, new_config)
    }
    while True:
        keep = {name for name in reusable if dependencies[name] <= reusable}
        if keep == reusable:
            break
        reusable = keep
    if not reusable:
        return None, set()
    state = {
        new[name].output_key: deepcopy(latest[name].state[new[name].output_key])
        for name in reusable
    }
    # Carry only provenance belonging to retained outputs; runtime budgets,
    # completion markers, rejected values and unknown agent state are not copied.
    from fedotmas.maw._file_evidence import FILE_EVIDENCE_STATE_KEY

    evidence: dict[str, Any] = {
        "agents": {},
        "artifacts": {},
        "source_aliases": {},
        "alias_owners": {},
    }
    for name in reusable:
        root = latest[name].state.get(FILE_EVIDENCE_STATE_KEY) or {}
        if name in (root.get("agents") or {}):
            evidence["agents"][name] = deepcopy(root["agents"][name])
        key = new[name].output_key
        if key in (root.get("artifacts") or {}):
            evidence["artifacts"][key] = root["artifacts"][key]
        for alias, owner in (root.get("alias_owners") or {}).items():
            if owner in reusable and alias in (root.get("source_aliases") or {}):
                evidence["source_aliases"][alias] = deepcopy(
                    root["source_aliases"][alias]
                )
                evidence["alias_owners"][alias] = owner
    if any(evidence.values()):
        state[FILE_EVIDENCE_STATE_KEY] = evidence
    from fedotmas.maw._calculation import CALCULATIONS_KEY

    calculations = {
        name: deepcopy(latest[name].state[CALCULATIONS_KEY][name])
        for name in reusable
        if name in (latest[name].state.get(CALCULATIONS_KEY) or {})
    }
    if calculations:
        state[CALCULATIONS_KEY] = calculations
    return state, reusable
