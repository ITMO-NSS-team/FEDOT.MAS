from __future__ import annotations

from google.adk.agents import LlmAgent
from google.adk.agents.base_agent import BaseAgent

from fedotmas.common.logging import get_logger
from fedotmas._settings import ModelConfig
from fedotmas.mas.models import MASConfig, MASAgentConfig
from fedotmas.maw.builder import _resolve_llm, frame_instruction
from fedotmas.mcp import MCPServerConfig, create_toolset

_log = get_logger("fedotmas.mas.builder")


def build_routing_system(
    config: MASConfig,
    *,
    mcp_registry: dict[str, MCPServerConfig] | None = None,
    worker_models: dict[str, ModelConfig] | None = None,
    autonomous: bool = True,
) -> BaseAgent:
    """Build an ADK agent tree with a coordinating parent agent.

    Workers use ADK's ``single_turn`` mode. In a sub-agent hierarchy this
    exposes each worker as a call-and-return tool instead of a transfer
    target, so the coordinator retains control after every delegation.
    """
    workers = []
    for w in config.workers:
        if not w.output_key:
            w = w.model_copy(update={"output_key": f"{w.name}_output"})
        workers.append(
            _build_routing_agent(
                w,
                mcp_registry,
                worker_models,
                autonomous=autonomous,
                mode="single_turn",
            )
        )

    # ADK registers ``single_turn`` children as call-and-return tools during
    # parent construction. Assigning them later leaves the coordinator's tool
    # registry unchanged.
    coord = _build_routing_agent(
        config.coordinator,
        mcp_registry,
        worker_models,
        autonomous=autonomous,
        sub_agents=workers,
    )

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
    *,
    autonomous: bool = True,
    mode: str | None = None,
    sub_agents: list[BaseAgent] | None = None,
) -> LlmAgent:
    tools: list = []
    for tool_name in cfg.tools:
        tools.append(create_toolset(tool_name, registry=mcp_registry))

    model = _resolve_llm(cfg.model, worker_models)
    _log.debug("Built routing agent | name={} model={}", cfg.name, model)

    agent_kwargs = dict(
        name=cfg.name,
        description=cfg.description,
        model=model,
        instruction=(
            frame_instruction(cfg.instruction) if autonomous else cfg.instruction
        ),
        output_key=cfg.output_key,
        tools=tools,
    )

    if mode is not None:
        agent_kwargs["mode"] = mode
    if sub_agents is not None:
        agent_kwargs["sub_agents"] = sub_agents

    return LlmAgent(**agent_kwargs)
