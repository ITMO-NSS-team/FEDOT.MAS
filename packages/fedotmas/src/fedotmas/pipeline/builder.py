from __future__ import annotations

from typing import TypeAlias

from google.adk.agents import LlmAgent, LoopAgent, ParallelAgent, SequentialAgent
from google.adk.agents.base_agent import BaseAgent, _SingleAgentCallback
from google.adk.models.base_llm import BaseLlm
from google.adk.tools.exit_loop_tool import exit_loop

from fedotmas.common.logging import get_logger
from fedotmas.config.settings import DEFAULT_META_MODEL, ModelConfig, get_max_loop_iterations
from fedotmas.llm import make_llm
from fedotmas.mcp import MCPServerConfig, create_toolset
from fedotmas.pipeline._ppline_utils import make_callbacks
from fedotmas.pipeline.models import AgentConfig, PipelineConfig, StepConfig

AgentCallback: TypeAlias = _SingleAgentCallback

_log = get_logger("fedotmas.pipeline.builder")


def build(
    config: PipelineConfig,
    *,
    mcp_registry: dict[str, MCPServerConfig] | None = None,
    worker_models: dict[str, ModelConfig] | None = None,
    before_agent_callbacks: list[_SingleAgentCallback] | None = None,
    after_agent_callbacks: list[_SingleAgentCallback] | None = None,
) -> BaseAgent:
    """Convert a ``PipelineConfig`` into an executable ADK agent tree."""
    agents_by_name: dict[str, AgentConfig] = {a.name: a for a in config.agents}
    return _build_node(
        config.pipeline, agents_by_name, mcp_registry, worker_models,
        before_agent_callbacks, after_agent_callbacks,
    )


def _build_node(
    node: StepConfig,
    agents: dict[str, AgentConfig],
    mcp_registry: dict[str, MCPServerConfig] | None,
    worker_models: dict[str, ModelConfig] | None,
    extra_before: list[_SingleAgentCallback] | None = None,
    extra_after: list[_SingleAgentCallback] | None = None,
) -> BaseAgent:
    if node.type == "agent":
        agent = _build_llm_agent(agents[node.agent_name], mcp_registry, worker_models)  # type: ignore[arg-type]
        _attach_callbacks(agent, extra_before, extra_after)
        return agent

    children = [
        _build_node(c, agents, mcp_registry, worker_models, extra_before, extra_after)
        for c in node.children
    ]

    if node.type == "sequential":
        name = _seq_name(children)
        _log.debug("Built sequential node | name={}", name)
        agent = SequentialAgent(name=name, sub_agents=children)
        _attach_callbacks(agent, extra_before, extra_after)
        return agent

    if node.type == "parallel":
        name = _par_name(children)
        _log.debug("Built parallel node | name={}", name)
        agent = ParallelAgent(name=name, sub_agents=children)
        _attach_callbacks(agent, extra_before, extra_after)
        return agent

    if node.type == "loop":
        # Inject exit_loop tool into the last sub-agent if it's an LlmAgent.
        _inject_exit_loop(children)
        max_iter = node.max_iterations or get_max_loop_iterations()
        _log.debug("Built loop node | max_iterations={}", max_iter)
        agent = LoopAgent(
            name=_loop_name(children),
            sub_agents=children,
            max_iterations=max_iter,
        )
        _attach_callbacks(agent, extra_before, extra_after)
        return agent

    raise ValueError(f"Unknown node type: {node.type}")


def _resolve_llm(
    model_name: str | None,
    worker_models: dict[str, ModelConfig] | None,
) -> str | BaseLlm:
    """Return a ``BaseLlm`` for known worker configs, else a plain model string.

    Model name normalization (provider prefix) is handled by
    ``AgentConfig`` model_validator, so *model_name* here is already
    normalized or ``None``.
    """
    if not model_name:
        _log.warning("No model specified for agent, using default: {}", DEFAULT_META_MODEL)
        return DEFAULT_META_MODEL
    if worker_models:
        cfg = worker_models.get(model_name)
        if cfg:
            return make_llm(cfg)
    return model_name


def _build_llm_agent(
    cfg: AgentConfig,
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


def _attach_callbacks(
    agent: BaseAgent,
    extra_before: list[_SingleAgentCallback] | None = None,
    extra_after: list[_SingleAgentCallback] | None = None,
) -> None:
    before_log, after_log = make_callbacks(agent.name)

    before_cbs: list[_SingleAgentCallback] = [before_log]
    if extra_before:
        before_cbs.extend(extra_before)

    after_cbs: list[_SingleAgentCallback] = [after_log]
    if extra_after:
        after_cbs.extend(extra_after)

    agent.before_agent_callback = before_cbs
    agent.after_agent_callback = after_cbs


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


def _seq_name(children: list[BaseAgent]) -> str:
    return "seq_" + "_".join(c.name for c in children)


def _par_name(children: list[BaseAgent]) -> str:
    return "par_" + "_".join(c.name for c in children)


def _loop_name(children: list[BaseAgent]) -> str:
    return "loop_" + "_".join(c.name for c in children)
