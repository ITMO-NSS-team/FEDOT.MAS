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

# -- Spinner frames --------------------------------------------------------
_SPINNER_FRAMES: tuple[str, ...] = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")

# -- Animation timing ------------------------------------------------------
_SPINNER_FPS = 8.0
_DONE_FLASH_DURATION = 0.6
_ERROR_FLASH_DURATION = 0.8


@dataclass
class _NodeState:
    label: str
    node_type: str = "agent"
    status: str = "pending"
    start_time: float = 0.0
    done_time: float = 0.0
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
        now = time.monotonic()

        if s.status == "running":
            if s.node_type == "agent":
                icon = _SPINNER_FRAMES[int(now * _SPINNER_FPS) % len(_SPINNER_FRAMES)]
            else:
                icon = "◌"
            style = "yellow"
            elapsed = f"  {now - s.start_time:.1f}s" if s.start_time else ""
            yield Text.from_markup(f"[{style}]{icon} {s.label}{elapsed}[/{style}]")

        elif s.status == "done":
            flash_age = now - s.done_time if s.done_time else _DONE_FLASH_DURATION + 1
            style = "bold green" if flash_age < _DONE_FLASH_DURATION else "green"
            elapsed = ""
            if s.start_time and s.done_time:
                elapsed = f"  {s.done_time - s.start_time:.1f}s"
            yield Text.from_markup(f"[{style}]✓ {s.label}{elapsed}[/{style}]")

        elif s.status == "error":
            flash_age = now - s.done_time if s.done_time else _ERROR_FLASH_DURATION + 1
            if flash_age < _ERROR_FLASH_DURATION:
                style = "bold red"
            else:
                style = "red"
            elapsed = ""
            if s.start_time and s.done_time:
                elapsed = f"  {s.done_time - s.start_time:.1f}s"
            yield Text.from_markup(f"[{style}]✗ {s.label}{elapsed}[/{style}]")

        else:  # pending
            icon, style = _STATUS_STYLE["pending"]
            yield Text.from_markup(f"[{style}]{icon} {s.label}[/{style}]")


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
        type_key = {"sequential": "seq", "parallel": "par", "loop": "loop"}.get(
            node.type, "agent"
        )
        state = _NodeState(label=label, node_type=type_key)
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

    def mark_done(self, name: str) -> None:
        state = self._nodes.get(name)
        if not state:
            return
        state.status = "done"
        state.done_time = time.monotonic()

    def mark_error(self, name: str) -> None:
        state = self._nodes.get(name)
        if not state:
            return
        state.status = "error"
        state.done_time = time.monotonic()

    @contextmanager
    def live(self) -> Generator[Self]:
        import loguru

        loguru.logger.disable("fedotmas")
        try:
            with Live(self._tree, refresh_per_second=8) as lv:
                self._live = lv
                yield self
                self._live = None
        finally:
            loguru.logger.enable("fedotmas")


def make_callbacks(viz: PipelineVisualizer, name: str) -> tuple[..., ...]:
    """Create before/after agent callbacks bound to *name*."""

    def before(*, callback_context, **_kw):  # noqa: ARG001
        viz.mark_running(name)
        return None

    def after(*, callback_context, **_kw):  # noqa: ARG001
        viz.mark_done(name)
        return None

    return before, after
