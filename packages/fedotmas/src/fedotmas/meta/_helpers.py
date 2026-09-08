from __future__ import annotations

from typing import Any, TypeVar

from pydantic import BaseModel

from fedotmas.maw.models import AgentPoolConfig
from fedotmas.mcp import MCPServerConfig, get_server_descriptions
from fedotmas._settings import (
    ModelConfig,
    get_meta_model,
    get_meta_temperature,
    get_worker_models,
    resolve_model_config,
)

T = TypeVar("T", bound=BaseModel)


def resolve_meta_and_workers(
    meta_model: str | ModelConfig | None,
    worker_models: list[str | ModelConfig] | None,
    temperature: float | None,
) -> tuple[ModelConfig, list[ModelConfig], float]:
    """Resolve meta model, worker models, and temperature with env fallback."""
    resolved_meta = (
        resolve_model_config(meta_model)
        if meta_model
        else resolve_model_config(get_meta_model())
    )
    resolved_workers = (
        [resolve_model_config(m) for m in worker_models]
        if worker_models
        else [resolve_model_config(m) for m in get_worker_models()]
    )
    resolved_temp = temperature if temperature is not None else get_meta_temperature()
    return resolved_meta, resolved_workers, resolved_temp


def format_server_descriptions(descriptions: dict[str, str]) -> str:
    """Format MCP server descriptions for LLM prompts."""
    if not descriptions:
        return "No MCP tools available."
    return "\n".join(f"- **{name}**: {desc}" for name, desc in descriptions.items())


def resolve_tool_descriptions(
    mcp_registry: dict[str, MCPServerConfig] | None,
    tool_catalog: dict[str, str] | None,
) -> dict[str, str]:
    """Return the ``{tool: description}`` catalogue to advertise to the meta-agent.

    An explicit *tool_catalog* wins over the local MCP registry, empty included:
    a config generated for another runtime has to offer that runtime's tools,
    not whatever this workspace happens to have discovered.
    """
    if tool_catalog is not None:
        return dict(tool_catalog)
    return get_server_descriptions(mcp_registry)


def format_agent_pool(pool: AgentPoolConfig) -> str:
    """Format an agent pool as readable text for a meta-agent user message."""
    lines: list[str] = []
    for a in pool.agents:
        parts = [f"- **{a.name}**"]
        if a.model:
            parts.append(f"  model: {a.model}")
        parts.append(f"  instruction: {a.instruction}")
        if a.tools:
            parts.append(f"  tools: {', '.join(a.tools)}")
        lines.append("\n".join(parts))
    return "\n\n".join(lines)


def parse_llm_output(raw: Any, schema: type[T]) -> T:
    """Parse raw LLM output (dict or JSON string) into a Pydantic model."""
    if isinstance(raw, dict):
        return schema.model_validate(raw)
    if isinstance(raw, str):
        return schema.model_validate_json(raw)
    raise TypeError(f"Unexpected LLM output type: {type(raw)}")


def validate_allowed_models(raw_output: Any, allowed_models: list[str]) -> None:
    """Raise ValueError if any agent.model is not in allowed_models."""
    if not allowed_models or not isinstance(raw_output, dict):
        return
    allowed = set(allowed_models)
    # MAW: raw_output["agents"] — list of agent dicts
    # MAS: raw_output["coordinator"] + raw_output["workers"]
    agents: list[dict] = []
    if "agents" in raw_output:
        agents = raw_output["agents"]
    else:
        if "coordinator" in raw_output:
            agents.append(raw_output["coordinator"])
        agents.extend(raw_output.get("workers", []))

    for agent in agents:
        model = agent.get("model")
        if model and model not in allowed:
            raise ValueError(
                f"Agent '{agent.get('name', '?')}' uses model '{model}' "
                f"not in allowed_models: {sorted(allowed)}"
            )
