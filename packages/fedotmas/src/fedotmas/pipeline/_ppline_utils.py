from __future__ import annotations

import io

from rich.console import Console
from rich.tree import Tree

from fedotmas.common.logging import get_logger
from fedotmas.pipeline.models import PipelineConfig, StepConfig

_log = get_logger("fedotmas.pipeline")

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


def _build_tree(
    node: StepConfig,
    parent: Tree,
    agents_by_name: dict[str, str],
) -> None:
    label = _node_label(node, agents_by_name)
    branch = parent.add(label)
    for child in node.children:
        _build_tree(child, branch, agents_by_name)


def print_tree(config: PipelineConfig) -> None:
    """Log the pipeline tree structure."""
    agents_by_name = {a.name: a.output_key for a in config.agents}
    tree = Tree("[bold]pipeline[/bold]")
    _build_tree(config.pipeline, tree, agents_by_name)
    buf = io.StringIO()
    Console(file=buf, highlight=False).print(tree)
    _log.info("Pipeline tree:\n{}", buf.getvalue().rstrip())
