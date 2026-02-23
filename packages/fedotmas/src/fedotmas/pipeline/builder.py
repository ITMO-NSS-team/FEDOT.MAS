from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from google.adk.agents import LlmAgent, LoopAgent, ParallelAgent, SequentialAgent
from google.adk.agents.base_agent import BaseAgent
from google.adk.tools.exit_loop_tool import exit_loop

from fedotmas.common.logging import get_logger
from fedotmas.config.settings import settings
from fedotmas.mcp import MCPServerConfig, create_toolset
from fedotmas.pipeline._ppline_utils import make_callbacks
from fedotmas.pipeline.models import AgentConfig, PipelineConfig, StepConfig

AgentCallback = Callable[..., Any]

_log = get_logger("fedotmas.pipeline.builder")


def build(
    config: PipelineConfig,
    *,
    mcp_registry: dict[str, MCPServerConfig] | None = None,
    before_agent_callbacks: list[AgentCallback] | None = None,
    after_agent_callbacks: list[AgentCallback] | None = None,
) -> BaseAgent:
    """Convert a ``PipelineConfig`` into an executable ADK agent tree."""
    agents_by_name: dict[str, AgentConfig] = {a.name: a for a in config.agents}
    return _build_node(
        config.pipeline, agents_by_name, mcp_registry,
        before_agent_callbacks, after_agent_callbacks,
    )


def _build_node(
    node: StepConfig,
    agents: dict[str, AgentConfig],
    mcp_registry: dict[str, MCPServerConfig] | None,
    extra_before: list[AgentCallback] | None = None,
    extra_after: list[AgentCallback] | None = None,
) -> BaseAgent:
    if node.type == "agent":
        agent = _build_llm_agent(agents[node.agent_name], mcp_registry)  # type: ignore[arg-type]
        _attach_callbacks(agent, extra_before, extra_after)
        return agent

    children = [
        _build_node(c, agents, mcp_registry, extra_before, extra_after)
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
        max_iter = node.max_iterations or settings.max_loop_iterations
        _log.debug("Built loop node | max_iterations={}", max_iter)
        agent = LoopAgent(
            name=_loop_name(children),
            sub_agents=children,
            max_iterations=max_iter,
        )
        _attach_callbacks(agent, extra_before, extra_after)
        return agent

    raise ValueError(f"Unknown node type: {node.type}")


_STATE_VAR_RE = re.compile(r"\{(\w+)\}")


def _make_vars_optional(instruction: str) -> str:
    """Convert ``{var}`` → ``{var?}`` so ADK treats missing state as empty string."""
    return _STATE_VAR_RE.sub(r"{\1?}", instruction)


def _resolve_model(model: str | None) -> str:
    """Resolve model name, adding LiteLLM provider prefix if missing."""
    resolved = model or settings.default_model
    if "/" not in resolved:
        resolved = f"openai/{resolved}"
        _log.debug("Added provider prefix | model={}", resolved)
    return resolved


def _build_llm_agent(
    cfg: AgentConfig,
    mcp_registry: dict[str, MCPServerConfig] | None,
) -> LlmAgent:
    tools: list = []
    for tool_name in cfg.tools:
        tools.append(create_toolset(tool_name, registry=mcp_registry))

    model = _resolve_model(cfg.model)
    _log.debug("Built agent | name={} model={}", cfg.name, model)
    return LlmAgent(
        name=cfg.name,
        model=model,
        instruction=_make_vars_optional(cfg.instruction),
        output_key=cfg.output_key,
        tools=tools,
    )


def _attach_callbacks(
    agent: BaseAgent,
    extra_before: list[AgentCallback] | None = None,
    extra_after: list[AgentCallback] | None = None,
) -> None:
    before_log, after_log = make_callbacks(agent.name)

    if extra_before:
        def before(*, callback_context, **kw):  # noqa: ARG001
            before_log(callback_context=callback_context, **kw)
            for cb in extra_before:
                cb(callback_context=callback_context, **kw)
            return None
    else:
        before = before_log

    if extra_after:
        def after(*, callback_context, **kw):  # noqa: ARG001
            after_log(callback_context=callback_context, **kw)
            for cb in extra_after:
                cb(callback_context=callback_context, **kw)
            return None
    else:
        after = after_log

    agent.before_agent_callback = before
    agent.after_agent_callback = after


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


def _seq_name(children: list[BaseAgent]) -> str:
    return "seq_" + "_".join(c.name for c in children)


def _par_name(children: list[BaseAgent]) -> str:
    return "par_" + "_".join(c.name for c in children)


def _loop_name(children: list[BaseAgent]) -> str:
    return "loop_" + "_".join(c.name for c in children)
