from __future__ import annotations

import os
from typing import Any

from google.adk.agents import LlmAgent, LoopAgent, ParallelAgent, SequentialAgent
from google.adk.agents.base_agent import BaseAgent
from google.adk.tools import FunctionTool
from google.adk.tools.exit_loop_tool import exit_loop
from google.adk.tools.mcp_tool import (
    McpToolset,
    StdioConnectionParams,
    StreamableHTTPConnectionParams,
)
from mcp import StdioServerParameters
from mcp.client.stdio import get_default_environment

from fedotmas._settings import ModelConfig
from fedotmas.common.llm import make_llm
from fedotmas.common.logging import get_logger
from fedotmas.interfaces.agent import (
    AgentDescriptor,
    AgentTree,
    LoopDescriptor,
    ParallelDescriptor,
    RoutingDescriptor,
    SequentialDescriptor,
)
from fedotmas.interfaces.tools import ToolDescriptor
from fedotmas.mcp._config import HttpMCPServer, StdioMCPServer

_log = get_logger("fedotmas.backends.adk.builder")


def build_adk_tree(tree: AgentTree) -> BaseAgent:
    """Convert a framework-neutral agent tree into an ADK agent tree."""
    if isinstance(tree, RoutingDescriptor):
        return _build_routing(tree)
    return _build_node(tree)


def _build_node(node: AgentTree) -> BaseAgent:
    if isinstance(node, AgentDescriptor):
        return _build_llm_agent(node)

    if isinstance(node, SequentialDescriptor):
        children = [_build_node(c) for c in node.children]
        return SequentialAgent(name=node.name, sub_agents=children)

    if isinstance(node, ParallelDescriptor):
        children = [_build_node(c) for c in node.children]
        return ParallelAgent(name=node.name, sub_agents=children)

    if isinstance(node, LoopDescriptor):
        children = [_build_node(c) for c in node.children]
        _inject_exit_loop(children, node.exit_loop_agent)
        return LoopAgent(
            name=node.name,
            sub_agents=children,
            max_iterations=node.max_iterations,
        )

    raise TypeError(f"Unknown agent tree node type: {type(node)}")


def _build_routing(desc: RoutingDescriptor) -> BaseAgent:
    """Build an ADK routing system using AutoFlow."""
    workers = [_build_llm_agent(w) for w in desc.workers]
    coord = _build_llm_agent(desc.coordinator)
    coord.sub_agents = workers
    _log.info(
        "Built ADK routing system | coordinator={} workers={}",
        coord.name,
        [w.name for w in workers],
    )
    return coord


def _build_llm_agent(desc: AgentDescriptor) -> LlmAgent:
    tools = _build_tools(desc.tools)
    model = _resolve_llm(desc.model)

    kwargs: dict[str, Any] = {
        "name": desc.name,
        "model": model,
        "instruction": desc.instruction,
        "output_key": desc.output_key,
        "tools": tools,
    }
    if desc.description:
        kwargs["description"] = desc.description
    if desc.temperature is not None:
        from google.genai import types

        kwargs["generate_content_config"] = types.GenerateContentConfig(
            temperature=desc.temperature,
        )

    _log.debug("Built ADK LlmAgent | name={} model={}", desc.name, model)
    return LlmAgent(**kwargs)


def _resolve_llm(model: str | ModelConfig | None) -> Any:
    """Convert model spec to an ADK-compatible model (BaseLlm or string)."""
    if model is None:
        from fedotmas._settings import DEFAULT_META_MODEL

        model = DEFAULT_META_MODEL
    if isinstance(model, str):
        from fedotmas._settings import resolve_model_config

        model = resolve_model_config(model)
    return make_llm(model)


def _build_tools(descriptors: list[ToolDescriptor]) -> list:
    """Convert ToolDescriptors into ADK tool objects."""
    tools: list = []
    for desc in descriptors:
        if desc.mcp_server is not None:
            tools.append(_build_mcp_toolset(desc))
        elif desc.function is not None:
            tool = FunctionTool(func=desc.function)
            tools.append(tool)
    return tools


def _build_mcp_toolset(desc: ToolDescriptor) -> McpToolset:
    """Create an ADK McpToolset from a ToolDescriptor."""
    cfg = desc.mcp_server
    if cfg is None:
        raise ValueError(f"ToolDescriptor '{desc.name}' has no mcp_server config")

    match cfg:
        case StdioMCPServer():
            env = {**get_default_environment(), **os.environ, **cfg.env}
            params = StdioConnectionParams(
                server_params=StdioServerParameters(
                    command=cfg.command,
                    args=list(cfg.args),
                    env=env,
                ),
                timeout=cfg.timeout,
            )
        case HttpMCPServer():
            params = StreamableHTTPConnectionParams(
                url=cfg.url,
                headers=cfg.headers or None,
                timeout=cfg.timeout,
            )
        case _:
            raise TypeError(f"Unsupported MCP server type: {type(cfg)}")

    return McpToolset(connection_params=params)


def _inject_exit_loop(
    children: list[BaseAgent], exit_loop_agent: str | None
) -> None:
    """Add ``exit_loop`` tool to the designated agent in a loop's children."""
    if exit_loop_agent is None:
        # Fall back to last LlmAgent
        for agent in reversed(children):
            if isinstance(agent, LlmAgent):
                _add_exit_loop(agent)
                break
        return

    for agent in children:
        if isinstance(agent, LlmAgent) and agent.name == exit_loop_agent:
            _add_exit_loop(agent)
            return


def _add_exit_loop(agent: LlmAgent) -> None:
    if agent.tools is None:
        agent.tools = [exit_loop]
    elif exit_loop not in agent.tools:
        agent.tools.append(exit_loop)  # type: ignore[arg-type]
    _log.debug("Injected exit_loop into agent={}", agent.name)
