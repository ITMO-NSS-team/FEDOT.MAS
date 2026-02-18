from __future__ import annotations

import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Self

from rich.console import Console, ConsoleOptions, RenderResult
from rich.live import Live
from rich.text import Text
from rich.tree import Tree

from fedotmas.pipeline.models import PipelineConfig, StepConfig

_WORKFLOW_ICONS: dict[str, str] = {
    "sequential": "→",
    "parallel": "‖",
    "loop": "↻",
}

_STATUS_STYLE: dict[str, tuple[str, str]] = {
    # status -> (icon, rich style)
    "pending": ("○", "dim"),
    "running": ("◌", "yellow"),
    "done": ("✓", "green"),
    "error": ("✗", "red"),
}


@dataclass
class _NodeState:
    label: str
    status: str = "pending"
    start_time: float = 0.0
    tree_node: Tree = field(default_factory=lambda: Tree(""))


class _DynamicLabel:
    """Renderable that recomputes elapsed time on every Live refresh."""

    __slots__ = ("_state",)

    def __init__(self, state: _NodeState) -> None:
        self._state = state

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        s = self._state
        icon, style = _STATUS_STYLE[s.status]
        elapsed = ""
        if s.start_time:
            elapsed = f"  {time.monotonic() - s.start_time:.1f}s"
        yield Text.from_markup(f"[{style}]{icon} {s.label}{elapsed}[/{style}]")


def _node_label(node: StepConfig, agents_by_name: dict[str, str]) -> str:
    if node.type == "agent":
        name = node.agent_name or "?"
        key = agents_by_name.get(name, "")
        return f"{name} \\[{key}]" if key else name
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
    """Real-time pipeline tree visualization using rich."""

    def __init__(self, config: PipelineConfig) -> None:
        self._nodes: dict[str, _NodeState] = {}
        agents_by_name = {a.name: a.output_key for a in config.agents}
        self._tree = Tree("[bold]pipeline[/bold]")
        self._build_tree(config.pipeline, self._tree, agents_by_name)
        self._live: Live | None = None

    def _build_tree(
        self,
        node: StepConfig,
        parent: Tree,
        agents_by_name: dict[str, str],
    ) -> None:
        nid = _node_id(node)
        label = _node_label(node, agents_by_name)
        state = _NodeState(label=label)
        dynamic = _DynamicLabel(state)
        branch = parent.add(dynamic)
        state.tree_node = branch
        self._nodes[nid] = state

        for child in node.children:
            self._build_tree(child, branch, agents_by_name)

    def mark_running(self, name: str) -> None:
        state = self._nodes.get(name)
        if not state:
            return
        state.status = "running"
        state.start_time = time.monotonic()
        if self._live:
            self._live.refresh()

    def mark_done(self, name: str) -> None:
        state = self._nodes.get(name)
        if not state:
            return
        state.status = "done"
        if self._live:
            self._live.refresh()

    def mark_error(self, name: str) -> None:
        state = self._nodes.get(name)
        if not state:
            return
        state.status = "error"
        if self._live:
            self._live.refresh()

    @contextmanager
    def live(self) -> Generator[Self]:
        import loguru

        loguru.logger.disable("fedotmas")
        try:
            with Live(self._tree, refresh_per_second=4) as lv:
                self._live = lv
                yield self
                self._live = None
        finally:
            loguru.logger.enable("fedotmas")


def make_callbacks(
    viz: PipelineVisualizer, name: str
) -> tuple[..., ...]:
    """Create before/after agent callbacks bound to *name*."""

    def before(*, callback_context, **_kw):  # noqa: ARG001
        viz.mark_running(name)
        return None

    def after(*, callback_context, **_kw):  # noqa: ARG001
        viz.mark_done(name)
        return None

    return before, after
