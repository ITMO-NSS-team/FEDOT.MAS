from __future__ import annotations

import uuid

from google.adk import Runner
from google.adk.agents import LlmAgent
from google.adk.sessions import InMemorySessionService
from google.genai import types

from pydantic import BaseModel

from fedotmas.common.logging import get_logger
from fedotmas.config.settings import settings
from fedotmas.mcp.registry import MCPServerConfig, get_server_descriptions
from fedotmas.meta.prompts import META_AGENT_SYSTEM_PROMPT
from fedotmas.meta.schema_utils import needs_strict_schema, patch_schema_openai_strict
from fedotmas.pipeline.models import PipelineConfig

_log = get_logger("fedotmas.meta.agent")


def _fix_schema_callback(*, callback_context, llm_request, **_kw):
    """Patch response_schema for OpenAI-compatible models (strict mode)."""
    model = llm_request.model or ""
    schema = llm_request.config and llm_request.config.response_schema
    if not schema or not needs_strict_schema(model):
        return None

    if isinstance(schema, type) and issubclass(schema, BaseModel):
        schema = schema.model_json_schema()
    elif isinstance(schema, BaseModel):
        schema = schema.__class__.model_json_schema()
    elif not isinstance(schema, dict):
        return None

    llm_request.config.response_schema = patch_schema_openai_strict(schema)
    return None


async def generate_pipeline_config(
    task: str,
    *,
    model: str | None = None,
    mcp_registry: dict[str, MCPServerConfig] | None = None,
) -> PipelineConfig:
    """Run the meta-agent and return a validated ``PipelineConfig``.

    Args:
        task: The user's task description.
        model: Override for the meta-agent model.
        mcp_registry: MCP server registry (defaults to the built-in one).

    Returns:
        A validated ``PipelineConfig`` ready for the builder.
    """
    descriptions = get_server_descriptions(mcp_registry)
    desc_text = _format_descriptions(descriptions)

    instruction = META_AGENT_SYSTEM_PROMPT.substitute(mcp_servers_desc=desc_text)

    resolved_model = model or settings.meta_agent_model
    resolved_temp = settings.meta_agent_temperature
    _log.info("Meta-agent | model={} temperature={}", resolved_model, resolved_temp)

    agent = LlmAgent(
        name="meta_agent",
        model=resolved_model,
        instruction=instruction,
        output_schema=PipelineConfig,
        output_key="pipeline_config",
        generate_content_config=types.GenerateContentConfig(
            temperature=resolved_temp,
        ),
        before_model_callback=_fix_schema_callback,
    )

    session_service = InMemorySessionService()
    session_id = uuid.uuid4().hex

    session = await session_service.create_session(
        app_name="fedotmas_meta",
        user_id="system",
        session_id=session_id,
    )

    runner = Runner(
        app_name="fedotmas_meta",
        agent=agent,
        session_service=session_service,
    )

    message = types.Content(
        role="user",
        parts=[types.Part.from_text(text=f"TASK: {task}")],
    )

    async for _event in runner.run_async(
        user_id="system",
        session_id=session.id,
        new_message=message,
    ):
        pass

    # Retrieve the structured output from session state.
    final_session = await session_service.get_session(
        app_name="fedotmas_meta",
        user_id="system",
        session_id=session.id,
    )

    raw_config = final_session.state.get("pipeline_config") if final_session else None
    if raw_config is None:
        raise RuntimeError(
            "Meta-agent did not produce a pipeline_config in session state"
        )

    # ADK stores the output_schema result as a dict (model_dump); validate it.
    if isinstance(raw_config, dict):
        config = PipelineConfig.model_validate(raw_config)
        _log.info("Config extracted | agents={}", len(config.agents))
        return config
    if isinstance(raw_config, str):
        config = PipelineConfig.model_validate_json(raw_config)
        _log.info("Config extracted (from JSON) | agents={}", len(config.agents))
        return config
    _log.error("Unexpected pipeline_config type: {}", type(raw_config))
    raise TypeError(f"Unexpected pipeline_config type: {type(raw_config)}")


def _format_descriptions(descriptions: dict[str, str]) -> str:
    if not descriptions:
        return "No MCP tools available."
    lines = []
    for name, desc in descriptions.items():
        lines.append(f"- **{name}**: {desc}")
    return "\n".join(lines)
