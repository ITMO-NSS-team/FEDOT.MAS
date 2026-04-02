from __future__ import annotations

from typing import Any, Optional

from google.adk.tools import BaseTool, ToolContext

from fedotmas.common.logging import get_logger
from fedotmas.maw._validators import (
    _find_terminal_node,
    collect_agent_refs,
)
from fedotmas.maw.models import MAWConfig

_log = get_logger("fedotmas.control.fixes.guardrails")


def run_config_guardrails(config: MAWConfig) -> list[str]:
    """Run guardrails on config. Returns list of errors (empty = valid).

    Reuses validators from ``maw/_validators.py`` but promotes warnings to
    errors.  Pydantic validators (unique names, unique output_keys, valid refs)
    already run during ``MAWConfig`` construction — this catches the softer
    checks.
    """
    errors: list[str] = []

    agent_names = {a.name for a in config.agents}
    referenced: set[str] = set()
    collect_agent_refs(config.pipeline, referenced)
    unused = agent_names - referenced
    if unused:
        errors.append(f"Unused agents not referenced in pipeline: {sorted(unused)}")

    terminal = _find_terminal_node(config.pipeline)
    if terminal.type == "parallel":
        errors.append("Pipeline ends with parallel node — add a synthesizer agent")

    return errors


async def guardrail_validate_config(
    tool: BaseTool,
    args: dict[str, Any],
    tool_context: ToolContext,
    tool_response: dict,
) -> Optional[dict]:
    """ADK ``after_tool_callback`` — auto-validates config after every fix tool.

    Returns ``None`` when config is valid (no override).  Returns an error dict
    when validation fails — the LLM sees the error and can retry.
    """
    config_raw = tool_context.state.get("config")
    if not config_raw:
        return None

    try:
        if isinstance(config_raw, str):
            config = MAWConfig.model_validate_json(config_raw)
        else:
            config = MAWConfig.model_validate(config_raw)
    except Exception as e:
        _log.warning("Config structurally invalid after {}: {}", tool.name, e)
        return {"error": f"Config structurally invalid after {tool.name}: {e}"}

    errors = run_config_guardrails(config)
    if errors:
        msg = "; ".join(errors)
        _log.warning("Guardrail failed after {}: {}", tool.name, msg)
        return {"error": f"Guardrail failed after {tool.name}: {msg}. Fix and retry."}

    return None
