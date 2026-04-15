from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union

from fedotmas._settings import ModelConfig
from fedotmas.interfaces.tools import ToolDescriptor


@dataclass
class AgentDescriptor:
    """Framework-neutral description of an LLM-driven agent."""

    name: str
    instruction: str
    model: str | ModelConfig | None = None
    tools: list[ToolDescriptor] = field(default_factory=list)
    output_key: str | None = None
    description: str = ""
    temperature: float | None = None


@dataclass
class SequentialDescriptor:
    """Ordered execution of child agents."""

    name: str
    children: list[AgentTree] = field(default_factory=list)


@dataclass
class ParallelDescriptor:
    """Concurrent execution of child agents."""

    name: str
    children: list[AgentTree] = field(default_factory=list)


@dataclass
class LoopDescriptor:
    """Repeated execution of child agents with exit control."""

    name: str
    children: list[AgentTree] = field(default_factory=list)
    max_iterations: int = 3
    exit_loop_agent: str | None = None
    """Name of the agent that should receive the exit_loop tool (if any)."""


@dataclass
class RoutingDescriptor:
    """Dynamic routing via a coordinator that dispatches to workers."""

    coordinator: AgentDescriptor
    workers: list[AgentDescriptor] = field(default_factory=list)


GroupDescriptor = Union[SequentialDescriptor, ParallelDescriptor, LoopDescriptor]
AgentTree = Union[AgentDescriptor, GroupDescriptor, RoutingDescriptor]
