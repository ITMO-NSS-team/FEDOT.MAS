from __future__ import annotations

from fedotmas.common.logging import get_logger
from fedotmas._settings import ModelConfig
from fedotmas.interfaces.agent import AgentDescriptor, RoutingDescriptor
from fedotmas.interfaces.tools import ToolDescriptor
from fedotmas.mas.models import MASConfig, MASAgentConfig
from fedotmas.maw.builder import _resolve_model
from fedotmas.mcp import MCPServerConfig, create_toolset

_log = get_logger("fedotmas.mas.builder")


def build_routing_system(
    config: MASConfig,
    *,
    mcp_registry: dict[str, MCPServerConfig] | None = None,
    worker_models: dict[str, ModelConfig] | None = None,
) -> RoutingDescriptor:
    """Build a framework-neutral routing descriptor with coordinator + workers."""
    workers = []
    for w in config.workers:
        if not w.output_key:
            w = w.model_copy(update={"output_key": f"{w.name}_output"})
        workers.append(_build_routing_agent(w, mcp_registry, worker_models))
    coord = _build_routing_agent(config.coordinator, mcp_registry, worker_models)
    _log.info(
        "Built routing descriptor | coordinator={} workers={}",
        coord.name,
        [w.name for w in workers],
    )
    return RoutingDescriptor(coordinator=coord, workers=workers)


def _build_routing_agent(
    cfg: MASAgentConfig,
    mcp_registry: dict[str, MCPServerConfig] | None,
    worker_models: dict[str, ModelConfig] | None,
) -> AgentDescriptor:
    tools: list[ToolDescriptor] = []
    for tool_name in cfg.tools:
        tools.append(create_toolset(tool_name, registry=mcp_registry))

    model = _resolve_model(cfg.model, worker_models)
    _log.debug("Built routing agent descriptor | name={} model={}", cfg.name, model)
    return AgentDescriptor(
        name=cfg.name,
        description=cfg.description,
        model=model,
        instruction=cfg.instruction,
        output_key=cfg.output_key,
        tools=tools,
    )
