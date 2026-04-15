from __future__ import annotations

import itertools
from typing import TypeAlias

from fedotmas.common.logging import get_logger
from fedotmas._settings import (
    DEFAULT_META_MODEL,
    ModelConfig,
    get_max_loop_iterations,
)
from fedotmas.interfaces.agent import (
    AgentDescriptor,
    AgentTree,
    LoopDescriptor,
    ParallelDescriptor,
    SequentialDescriptor,
)
from fedotmas.interfaces.tools import ToolDescriptor
from fedotmas.mcp import MCPServerConfig, create_toolset
from fedotmas.maw.models import MAWAgentConfig, MAWConfig, MAWStepConfig

_log = get_logger("fedotmas.maw.builder")


def build(
    config: MAWConfig,
    *,
    mcp_registry: dict[str, MCPServerConfig] | None = None,
    worker_models: dict[str, ModelConfig] | None = None,
) -> AgentTree:
    """Convert a ``MAWConfig`` into a framework-neutral agent tree."""
    agents_by_name: dict[str, MAWAgentConfig] = {a.name: a for a in config.agents}
    return _build_node(
        config.pipeline,
        agents_by_name,
        mcp_registry,
        worker_models,
    )


def _build_node(
    node: MAWStepConfig,
    agents: dict[str, MAWAgentConfig],
    mcp_registry: dict[str, MCPServerConfig] | None,
    worker_models: dict[str, ModelConfig] | None,
) -> AgentTree:
    if node.type == "agent":
        if node.agent_name is None:
            raise ValueError(f"Agent node missing 'agent_name': {node}")
        return _build_agent_descriptor(agents[node.agent_name], mcp_registry, worker_models)

    children = [
        _build_node(c, agents, mcp_registry, worker_models) for c in node.children
    ]

    if node.type == "sequential":
        name = _seq_name(children)
        _log.debug("Built sequential node | name={}", name)
        return SequentialDescriptor(name=name, children=children)

    if node.type == "parallel":
        name = _par_name(children)
        _log.debug("Built parallel node | name={}", name)
        return ParallelDescriptor(name=name, children=children)

    if node.type == "loop":
        max_iter = node.max_iterations or get_max_loop_iterations()
        exit_agent = _find_exit_loop_agent(children)
        _log.debug("Built loop node | max_iterations={}", max_iter)
        return LoopDescriptor(
            name=_loop_name(children),
            children=children,
            max_iterations=max_iter,
            exit_loop_agent=exit_agent,
        )

    raise ValueError(f"Unknown node type: {node.type}")


def _resolve_model(
    model_name: str | None,
    worker_models: dict[str, ModelConfig] | None,
) -> str | ModelConfig | None:
    """Return a ``ModelConfig`` for known worker configs, else a plain model string."""
    if not model_name:
        _log.warning(
            "No model specified for agent, using default: {}", DEFAULT_META_MODEL
        )
        return DEFAULT_META_MODEL
    if worker_models:
        cfg = worker_models.get(model_name)
        if cfg:
            return cfg
    return model_name


def _build_agent_descriptor(
    cfg: MAWAgentConfig,
    mcp_registry: dict[str, MCPServerConfig] | None,
    worker_models: dict[str, ModelConfig] | None,
) -> AgentDescriptor:
    tools: list[ToolDescriptor] = []
    for tool_name in cfg.tools:
        tools.append(create_toolset(tool_name, registry=mcp_registry))

    model = _resolve_model(cfg.model, worker_models)
    _log.debug("Built agent descriptor | name={} model={}", cfg.name, model)
    return AgentDescriptor(
        name=cfg.name,
        model=model,
        instruction=cfg.instruction,
        output_key=cfg.output_key,
        tools=tools,
    )


def _find_exit_loop_agent(children: list[AgentTree]) -> str | None:
    """Find the name of the last AgentDescriptor in children for exit_loop injection."""
    for child in reversed(children):
        if isinstance(child, AgentDescriptor):
            _log.debug("exit_loop target agent={}", child.name)
            return child.name
    return None


WORKFLOW_PREFIXES = ("seq_", "par_", "loop_")


_node_counter = itertools.count(1)


def _next_id() -> int:
    return next(_node_counter)


def _seq_name(_children: list[AgentTree]) -> str:
    return f"seq_{_next_id()}"


def _par_name(_children: list[AgentTree]) -> str:
    return f"par_{_next_id()}"


def _loop_name(_children: list[AgentTree]) -> str:
    return f"loop_{_next_id()}"
