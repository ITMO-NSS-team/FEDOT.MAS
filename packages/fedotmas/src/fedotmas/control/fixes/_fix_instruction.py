from __future__ import annotations

from google.adk.tools import ToolContext

from fedotmas.common.logging import get_logger
from fedotmas.maw.models import MAWConfig

_log = get_logger("fedotmas.control.fixes.fix_instruction")


def _load_config(raw: str | dict) -> MAWConfig:
    if isinstance(raw, str):
        return MAWConfig.model_validate_json(raw)
    return MAWConfig.model_validate(raw)


async def fix_instruction(
    tool_context: ToolContext,
    agent_name: str,
    new_instruction: str,
    reasoning: str,
) -> str:
    """Rewrite a failing agent's instruction prompt to fix the error.

    Args:
        tool_context: ADK tool context (injected automatically).
        agent_name: Name of the agent to fix.
        new_instruction: Complete rewritten instruction for the agent.
        reasoning: Brief explanation of what was wrong and how this fixes it.
    """
    config = _load_config(tool_context.state["config"])
    agent = next((a for a in config.agents if a.name == agent_name), None)
    if agent is None:
        names = [a.name for a in config.agents]
        return f"Error: agent '{agent_name}' not found. Available: {names}"

    new_agent = agent.model_copy(update={"instruction": new_instruction})
    try:
        new_config = config.replace_agent(agent_name, new_agent)
    except Exception as e:
        return f"Error applying fix: {e}"

    tool_context.state["config"] = new_config.model_dump_json()
    _log.info(
        "Fixed instruction for '{}' | reason: {}",
        agent_name,
        reasoning,
    )
    return f"Updated instruction for '{agent_name}'. Reason: {reasoning}"
