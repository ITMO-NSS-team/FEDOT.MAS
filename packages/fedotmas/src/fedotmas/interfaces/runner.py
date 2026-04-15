from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from fedotmas._settings import ModelConfig
from fedotmas.interfaces.agent import AgentTree
from fedotmas.interfaces.middleware import MiddlewareProtocol
from fedotmas.interfaces.tools import ToolDescriptor


@dataclass
class PipelineResult:
    """Result of a pipeline execution."""

    state: dict[str, Any] = field(default_factory=dict)
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    elapsed: float = 0.0


@dataclass
class SingleAgentResult:
    """Result of a single-agent call (meta-agent, debugger, etc.)."""

    raw_output: Any
    state: dict[str, Any] = field(default_factory=dict)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    elapsed: float = 0.0


@runtime_checkable
class RunnerProtocol(Protocol):
    """Backend-agnostic runner for executing agent trees."""

    async def run_pipeline(
        self,
        agent_tree: AgentTree,
        user_query: str,
        *,
        initial_state: dict[str, Any] | None = None,
        middlewares: list[MiddlewareProtocol] | None = None,
        backend_plugins: list[Any] | None = None,
    ) -> PipelineResult:
        """Execute a full agent tree and return the final state."""
        ...

    async def run_single_agent(
        self,
        *,
        agent_name: str,
        instruction: str,
        user_message: str,
        model: str | ModelConfig,
        temperature: float,
        output_schema: type[BaseModel] | None = None,
        output_key: str,
        tools: list[ToolDescriptor] | None = None,
        after_tool_callback: Any | None = None,
        initial_state: dict[str, Any] | None = None,
        backend_plugins: list[Any] | None = None,
    ) -> SingleAgentResult:
        """Execute a single LLM agent call (for meta-agent, debugger, etc.)."""
        ...
