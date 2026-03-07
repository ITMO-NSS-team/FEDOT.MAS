from __future__ import annotations

from google.adk.agents import LlmAgent
from google.adk.agents.base_agent import BaseAgent

from fedotmas.common.logging import get_logger
from fedotmas._settings import ModelConfig
from fedotmas.mas.models import MASConfig, MASAgentConfig
from fedotmas.maw.builder import _resolve_llm
from fedotmas.mcp import MCPServerConfig, create_toolset

_log = get_logger("fedotmas.mas.builder")


def build_routing_system(
    config: MASConfig,
    *,
    mcp_registry: dict[str, MCPServerConfig] | None = None,
    worker_models: dict[str, ModelConfig] | None = None,
) -> BaseAgent:
    """Build an ADK agent tree with LLM-driven routing via AutoFlow.

    The coordinator agent gets workers as ``sub_agents``, which enables
    ADK AutoFlow's ``transfer_to_agent`` mechanism for dynamic routing.
    """
    workers = [
        _build_routing_agent(w, mcp_registry, worker_models) for w in config.workers
    ]
    coord = _build_routing_agent(config.coordinator, mcp_registry, worker_models)
    coord.sub_agents = workers  # ADK AutoFlow activates automatically
    _log.info(
        "Built routing system | coordinator={} workers={}",
        coord.name,
        [w.name for w in workers],
    )
    return coord


def _build_routing_agent(
    cfg: MASAgentConfig,
    mcp_registry: dict[str, MCPServerConfig] | None,
    worker_models: dict[str, ModelConfig] | None,
) -> LlmAgent:
    tools: list = []
    for tool_name in cfg.tools:
        tools.append(create_toolset(tool_name, registry=mcp_registry))

    model = _resolve_llm(cfg.model, worker_models)
    _log.debug("Built routing agent | name={} model={}", cfg.name, model)
    return LlmAgent(
        name=cfg.name,
        description=cfg.description,
        model=model,
        instruction=cfg.instruction,
        output_key=cfg.output_key,
        tools=tools,
    )
