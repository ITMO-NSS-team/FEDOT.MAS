from __future__ import annotations

import uuid

from google.adk import Runner
from google.adk.agents import LlmAgent
from google.adk.sessions import InMemorySessionService
from google.genai import types

from fedotmas.config.settings import settings
from fedotmas.mcp.registry import MCPServerConfig, get_server_descriptions
from fedotmas.meta.prompts import META_AGENT_SYSTEM_PROMPT
from fedotmas.pipeline.models import PipelineConfig


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

    agent = LlmAgent(
        name="meta_agent",
        model=model or settings.meta_agent_model,
        instruction=instruction,
        output_schema=PipelineConfig,
        output_key="pipeline_config",
        generate_content_config=types.GenerateContentConfig(
            temperature=settings.meta_agent_temperature,
        ),
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
        return PipelineConfig.model_validate(raw_config)
    if isinstance(raw_config, str):
        return PipelineConfig.model_validate_json(raw_config)
    raise TypeError(f"Unexpected pipeline_config type: {type(raw_config)}")


def _format_descriptions(descriptions: dict[str, str]) -> str:
    if not descriptions:
        return "No MCP tools available."
    lines = []
    for name, desc in descriptions.items():
        lines.append(f"- **{name}**: {desc}")
    return "\n".join(lines)
