from __future__ import annotations

from typing import TYPE_CHECKING

from fedotmas.common.logging import get_logger

if TYPE_CHECKING:
    from fedotmas.maw.models import MAWStepConfig

_log = get_logger("fedotmas.maw.validators")


def auto_fill_agent_name(node: MAWStepConfig, name: str) -> None:
    """Fill missing agent_name when there is exactly one agent."""
    if node.type == "agent" and node.agent_name is None:
        node.agent_name = name
    for child in node.children:
        auto_fill_agent_name(child, name)


def collect_agent_refs(node: MAWStepConfig, refs: set[str]) -> None:
    if node.type == "agent" and node.agent_name:
        refs.add(node.agent_name)
    for child in node.children:
        collect_agent_refs(child, refs)


def validate_node_refs(node: MAWStepConfig, agent_names: set[str]) -> None:
    if node.type == "agent":
        if node.agent_name is None:
            raise ValueError("Node of type 'agent' must have an agent_name")
        if node.agent_name not in agent_names:
            raise ValueError(
                f"Pipeline references unknown agent '{node.agent_name}'. "
                f"Available: {sorted(agent_names)}"
            )
    else:
        if not node.children:
            raise ValueError(f"Node of type '{node.type}' must have at least one child")
        for child in node.children:
            validate_node_refs(child, agent_names)


def warn_unused_agents(pipeline: MAWStepConfig, agent_names: set[str]) -> None:
    referenced: set[str] = set()
    collect_agent_refs(pipeline, referenced)
    unused = agent_names - referenced
    if unused:
        _log.warning("Unused agents: {}", sorted(unused))


def _find_terminal_node(node: MAWStepConfig) -> MAWStepConfig:
    """Return the terminal (last-executing) node in the pipeline tree."""
    if node.type in ("agent", "parallel"):
        return node
    if node.type in ("sequential", "loop") and node.children:
        return _find_terminal_node(node.children[-1])
    return node


def warn_terminal_parallel(pipeline: MAWStepConfig) -> None:
    """Warn if the pipeline ends with a parallel node (no synthesizer)."""
    terminal = _find_terminal_node(pipeline)
    if terminal.type == "parallel":
        _log.warning(
            "Pipeline ends with a parallel node — results will not be "
            "synthesized. Add a synthesizer agent after the parallel block."
        )
