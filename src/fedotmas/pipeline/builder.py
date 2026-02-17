"""Build an ADK agent tree from a PipelineConfig."""

from __future__ import annotations

from google.adk.agents import LlmAgent, LoopAgent, ParallelAgent, SequentialAgent
from google.adk.agents.base_agent import BaseAgent
from google.adk.tools.exit_loop_tool import exit_loop

from fedotmas.config.settings import DEFAULT_MAX_LOOP_ITERATIONS, DEFAULT_MODEL
from fedotmas.mcp.registry import MCPServerConfig, create_toolset
from fedotmas.pipeline.models import AgentConfig, PipelineConfig, PipelineNodeConfig


def build(
    config: PipelineConfig,
    *,
    mcp_registry: dict[str, MCPServerConfig] | None = None,
) -> BaseAgent:
    """Convert a ``PipelineConfig`` into an executable ADK agent tree.

    Returns the root ``BaseAgent`` ready for ``Runner.run_async``.
    """
    agents_by_name: dict[str, AgentConfig] = {a.name: a for a in config.agents}
    return _build_node(config.pipeline, agents_by_name, mcp_registry)


def _build_node(
    node: PipelineNodeConfig,
    agents: dict[str, AgentConfig],
    mcp_registry: dict[str, MCPServerConfig] | None,
) -> BaseAgent:
    if node.type == "agent":
        return _build_llm_agent(agents[node.agent_name], mcp_registry)  # type: ignore[arg-type]

    children = [_build_node(c, agents, mcp_registry) for c in node.children]

    if node.type == "sequential":
        return SequentialAgent(name=_seq_name(children), sub_agents=children)

    if node.type == "parallel":
        return ParallelAgent(name=_par_name(children), sub_agents=children)

    if node.type == "loop":
        # Inject exit_loop tool into the last sub-agent if it's an LlmAgent.
        _inject_exit_loop(children)
        max_iter = node.max_iterations or DEFAULT_MAX_LOOP_ITERATIONS
        return LoopAgent(
            name=_loop_name(children),
            sub_agents=children,
            max_iterations=max_iter,
        )

    raise ValueError(f"Unknown node type: {node.type}")


# ------------------------------------------------------------------
# LlmAgent construction
# ------------------------------------------------------------------


def _build_llm_agent(
    cfg: AgentConfig,
    mcp_registry: dict[str, MCPServerConfig] | None,
) -> LlmAgent:
    tools: list = []
    for tool_name in cfg.tools:
        tools.append(create_toolset(tool_name, registry=mcp_registry))

    return LlmAgent(
        name=cfg.name,
        model=cfg.model or DEFAULT_MODEL,
        instruction=cfg.instruction,
        output_key=cfg.output_key,
        tools=tools,
    )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _inject_exit_loop(children: list[BaseAgent]) -> None:
    """Add ``exit_loop`` tool to the last LlmAgent in a loop's children."""
    for agent in reversed(children):
        if isinstance(agent, LlmAgent):
            if agent.tools is None:
                agent.tools = [exit_loop]
            elif exit_loop not in agent.tools:
                agent.tools.append(exit_loop)
            break


def _seq_name(children: list[BaseAgent]) -> str:
    return "seq_" + "_".join(c.name for c in children)


def _par_name(children: list[BaseAgent]) -> str:
    return "par_" + "_".join(c.name for c in children)


def _loop_name(children: list[BaseAgent]) -> str:
    return "loop_" + "_".join(c.name for c in children)
