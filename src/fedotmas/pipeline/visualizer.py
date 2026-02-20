from __future__ import annotations

import time

from rich.console import Console
from rich.tree import Tree

from fedotmas.common.logging import get_logger
from fedotmas.pipeline.models import PipelineConfig, StepConfig

_log = get_logger("fedotmas.pipeline.visualizer")

_WORKFLOW_ICONS: dict[str, str] = {
    "sequential": "→",
    "parallel": "‖",
    "loop": "↻",
}


def _node_label(node: StepConfig, agents_by_name: dict[str, str]) -> str:
    if node.type == "agent":
        name = node.agent_name or "?"
        key = agents_by_name.get(name, "")
        return f"{name} [{key}]" if key else name
    icon = _WORKFLOW_ICONS.get(node.type, "?")
    suffix = f" (max={node.max_iterations})" if node.max_iterations else ""
    return f"{icon} {node.type}{suffix}"


def _node_id(node: StepConfig) -> str:
    """Stable identifier matching the agent name builder.py will assign."""
    if node.type == "agent":
        return node.agent_name or "?"
    child_names = [_node_id(c) for c in node.children]
    prefix = {"sequential": "seq", "parallel": "par", "loop": "loop"}
    return f"{prefix.get(node.type, node.type)}_{'_'.join(child_names)}"


class PipelineVisualizer:
    """Pipeline tree visualizer: prints static tree and logs agent lifecycle."""

    def __init__(self, config: PipelineConfig) -> None:
        self._start_times: dict[str, float] = {}
        agents_by_name = {a.name: a.output_key for a in config.agents}
        self._tree = Tree("[bold]pipeline[/bold]")
        self._build_tree(config.pipeline, self._tree, agents_by_name)

    def _build_tree(
        self,
        node: StepConfig,
        parent: Tree,
        agents_by_name: dict[str, str],
    ) -> None:
        label = _node_label(node, agents_by_name)
        branch = parent.add(label)
        for child in node.children:
            self._build_tree(child, branch, agents_by_name)

    def print_tree(self) -> None:
        """Print the pipeline tree once to the console."""
        Console().print(self._tree)

    def mark_running(self, name: str) -> None:
        self._start_times[name] = time.monotonic()
        _log.info("Agent started | name={}", name)

    def mark_done(self, name: str) -> None:
        elapsed = time.monotonic() - self._start_times.get(name, time.monotonic())
        _log.info("Agent done | name={} elapsed={:.1f}s", name, elapsed)

    def mark_error(self, name: str) -> None:
        elapsed = time.monotonic() - self._start_times.get(name, time.monotonic())
        _log.error("Agent error | name={} elapsed={:.1f}s", name, elapsed)


def make_callbacks(viz: PipelineVisualizer, name: str) -> tuple[..., ...]:
    """Create before/after agent callbacks bound to *name*."""

    def before(*, callback_context, **_kw):  # noqa: ARG001
        viz.mark_running(name)
        return None

    def after(*, callback_context, **_kw):  # noqa: ARG001
        viz.mark_done(name)
        return None

    return before, after
