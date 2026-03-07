from __future__ import annotations

from typing import TypeAlias

from google.adk.agents import LlmAgent, LoopAgent, ParallelAgent, SequentialAgent
from google.adk.agents.base_agent import BaseAgent
from google.adk.models.base_llm import BaseLlm
from google.adk.tools.exit_loop_tool import exit_loop

from fedotmas.common.logging import get_logger
from fedotmas._settings import (
    DEFAULT_META_MODEL,
    ModelConfig,
    get_max_loop_iterations,
)
from fedotmas.common.llm import make_llm
from fedotmas.mcp import MCPServerConfig, create_toolset
from fedotmas.maw.models import MAWAgentConfig, MAWConfig, MAWStepConfig

AgentTree: TypeAlias = BaseAgent

_log = get_logger("fedotmas.maw.builder")


def build(
    config: MAWConfig,
    *,
    mcp_registry: dict[str, MCPServerConfig] | None = None,
    worker_models: dict[str, ModelConfig] | None = None,
) -> BaseAgent:
    """Convert a ``MAWConfig`` into an executable ADK agent tree."""
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
) -> BaseAgent:
    if node.type == "agent":
        if node.agent_name is None:
            raise ValueError(f"Agent node missing 'agent_name': {node}")
        return _build_llm_agent(agents[node.agent_name], mcp_registry, worker_models)

    children = [
        _build_node(c, agents, mcp_registry, worker_models) for c in node.children
    ]

    if node.type == "sequential":
        name = _seq_name(children)
        _log.debug("Built sequential node | name={}", name)
        return SequentialAgent(name=name, sub_agents=children)

    if node.type == "parallel":
        name = _par_name(children)
        _log.debug("Built parallel node | name={}", name)
        return ParallelAgent(name=name, sub_agents=children)

    if node.type == "loop":
        # Inject exit_loop tool into the last sub-agent if it's an LlmAgent.
        _inject_exit_loop(children)
        max_iter = node.max_iterations or get_max_loop_iterations()
        _log.debug("Built loop node | max_iterations={}", max_iter)
        return LoopAgent(
            name=_loop_name(children),
            sub_agents=children,
            max_iterations=max_iter,
        )

    raise ValueError(f"Unknown node type: {node.type}")


def _resolve_llm(
    model_name: str | None,
    worker_models: dict[str, ModelConfig] | None,
) -> str | BaseLlm:
    """Return a ``BaseLlm`` for known worker configs, else a plain model string.

    Model name normalization (provider prefix) is handled by
    ``MAWAgentConfig`` model_validator, so *model_name* here is already
    normalized or ``None``.
    """
    if not model_name:
        _log.warning(
            "No model specified for agent, using default: {}", DEFAULT_META_MODEL
        )
        return DEFAULT_META_MODEL
    if worker_models:
        cfg = worker_models.get(model_name)
        if cfg:
            return make_llm(cfg)
    return model_name


def _build_llm_agent(
    cfg: MAWAgentConfig,
    mcp_registry: dict[str, MCPServerConfig] | None,
    worker_models: dict[str, ModelConfig] | None,
) -> LlmAgent:
    tools: list = []
    for tool_name in cfg.tools:
        tools.append(create_toolset(tool_name, registry=mcp_registry))

    model = _resolve_llm(cfg.model, worker_models)
    _log.debug("Built agent | name={} model={}", cfg.name, model)
    return LlmAgent(
        name=cfg.name,
        model=model,
        instruction=cfg.instruction,
        output_key=cfg.output_key,
        tools=tools,
    )


def _inject_exit_loop(children: list[BaseAgent]) -> None:
    """Add ``exit_loop`` tool to the last LlmAgent in a loop's children."""
    for agent in reversed(children):
        if isinstance(agent, LlmAgent):
            if agent.tools is None:
                agent.tools = [exit_loop]
            elif exit_loop not in agent.tools:
                agent.tools.append(exit_loop)  # type: ignore[arg-type]
            _log.debug("Injected exit_loop into agent={}", agent.name)
            break


WORKFLOW_PREFIXES = ("seq_", "par_", "loop_")


_node_counter = 0


def _next_id() -> int:
    global _node_counter  # noqa: PLW0603
    _node_counter += 1
    return _node_counter


def _seq_name(_children: list[BaseAgent]) -> str:
    return f"seq_{_next_id()}"


def _par_name(_children: list[BaseAgent]) -> str:
    return f"par_{_next_id()}"


def _loop_name(_children: list[BaseAgent]) -> str:
    return f"loop_{_next_id()}"
