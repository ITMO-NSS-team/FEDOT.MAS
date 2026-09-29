"""Select the configured final output independently of runtime diagnostics."""

from __future__ import annotations

import json
from typing import Any

from fedotmas import MASConfig, MAWConfig
from fedotmas.maw._validators import _find_terminal_node


def prepare_output_key(config: MASConfig | MAWConfig) -> str | None:
    """Ensure MAS records its coordinator output; locate the MAW terminal agent."""
    if isinstance(config, MASConfig):
        if not config.coordinator.output_key:
            used = {
                worker.output_key or f"{worker.name}_output"
                for worker in config.workers
            }
            key = "coordinator_result"
            while key in used:
                key += "_"
            config.coordinator.output_key = key
        return config.coordinator.output_key
    terminal = _find_terminal_node(config.pipeline)
    name = config.final_answer_agent or (
        terminal.agent_name if terminal.type == "agent" else None
    )
    return next(
        (agent.output_key for agent in config.agents if agent.name == name), None
    )


def result_payload(state: dict[str, Any], output_key: str | None, status: str) -> dict:
    value = state.get(output_key) if output_key is not None else None
    answer = (
        value
        if isinstance(value, str)
        else json.dumps(value, ensure_ascii=False, default=str)
        if value is not None
        else ""
    )
    if status == "completed" and not answer.strip():
        status = "incomplete"
    return {"answer": answer, "output_key": output_key, "status": status}
