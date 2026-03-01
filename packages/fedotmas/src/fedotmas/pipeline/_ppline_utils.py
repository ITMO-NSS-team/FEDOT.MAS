from __future__ import annotations

import time

from rich.console import Console
from rich.tree import Tree

from google.adk.agents.base_agent import _SingleAgentCallback
from google.adk.agents.callback_context import CallbackContext

from fedotmas.common.logging import get_logger
from fedotmas.pipeline.models import PipelineConfig, StepConfig

_log = get_logger("fedotmas.pipeline")

_WORKFLOW_ICONS: dict[str, str] = {
    "sequential": "→",
    "parallel": "‖",
    "loop": "↻",
}

_start_times: dict[str, float] = {}


def _node_label(node: StepConfig, agents_by_name: dict[str, str]) -> str:
    if node.type == "agent":
        name = node.agent_name or "?"
        key = agents_by_name.get(name, "")
        return f"{name} [{key}]" if key else name
    icon = _WORKFLOW_ICONS.get(node.type, "?")
    suffix = f" (max={node.max_iterations})" if node.max_iterations else ""
    return f"{icon} {node.type}{suffix}"


def _build_tree(
    node: StepConfig,
    parent: Tree,
    agents_by_name: dict[str, str],
) -> None:
    label = _node_label(node, agents_by_name)
    branch = parent.add(label)
    for child in node.children:
        _build_tree(child, branch, agents_by_name)


def _is_workflow_node(name: str) -> bool:
    from fedotmas.pipeline.builder import WORKFLOW_PREFIXES

    return name.startswith(WORKFLOW_PREFIXES)


def print_tree(config: PipelineConfig) -> None:
    """Print the pipeline tree once to the console."""
    agents_by_name = {a.name: a.output_key for a in config.agents}
    tree = Tree("[bold]pipeline[/bold]")
    _build_tree(config.pipeline, tree, agents_by_name)
    Console().print(tree)


def _mark_running(name: str) -> None:
    _start_times[name] = time.monotonic()
    if not _is_workflow_node(name):
        _log.info("Agent started | name={}", name)


def _mark_done(name: str) -> None:
    elapsed = time.monotonic() - _start_times.pop(name, time.monotonic())
    if not _is_workflow_node(name):
        _log.info("Agent done | name={} elapsed={:.1f}s", name, elapsed)


def make_callbacks(name: str) -> tuple[_SingleAgentCallback, _SingleAgentCallback]:
    """Create before/after agent callbacks that log agent lifecycle."""

    def before(callback_context: CallbackContext) -> None:  # noqa: ARG001
        _mark_running(name)

    def after(callback_context: CallbackContext) -> None:  # noqa: ARG001
        _mark_done(name)

    return before, after
