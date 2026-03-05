from __future__ import annotations

from typing import Union

from fastapi import FastAPI
from google.adk.agents.base_agent import BaseAgent
from google.adk.apps.app import App
from google.adk.cli.fast_api import get_fast_api_app
from google.adk.cli.utils.base_agent_loader import BaseAgentLoader


class MASAgentLoader(BaseAgentLoader):
    """Agent loader that serves pre-built BaseAgent instances.

    Use this instead of ADK's filesystem-based AgentLoader when agents
    are built programmatically (e.g. via MAS().build()).
    """

    def __init__(self) -> None:
        self._agents: dict[str, BaseAgent | App] = {}

    def register(self, name: str, agent: BaseAgent | App) -> None:
        """Register an agent or App under a given name."""
        self._agents[name] = agent

    def load_agent(self, agent_name: str) -> Union[BaseAgent, App]:
        if agent_name not in self._agents:
            available = ", ".join(sorted(self._agents)) or "(none)"
            raise KeyError(
                f"Agent '{agent_name}' not registered. Available: {available}"
            )
        return self._agents[agent_name]

    def list_agents(self) -> list[str]:
        return sorted(self._agents)


def create_api_app(
    agents: dict[str, BaseAgent | App],
    *,
    session_service_uri: str | None = None,
    memory_service_uri: str | None = None,
    artifact_service_uri: str | None = None,
    web: bool = False,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> FastAPI:
    """Create a FastAPI app that serves MAS-generated agents via ADK API server.

    Args:
        agents: Mapping of agent name to BaseAgent or App instance.
        session_service_uri: URI for session backend (e.g. "sqlite:///sessions.db",
            "postgresql://..."). Defaults to in-memory.
        memory_service_uri: URI for memory service. Defaults to None.
        artifact_service_uri: URI for artifact storage. Defaults to None.
        web: If True, include the ADK web UI.
        host: Bind address.
        port: Bind port.
    """
    loader = MASAgentLoader()
    for name, agent in agents.items():
        loader.register(name, agent)

    return get_fast_api_app(
        agents_dir=".",
        agent_loader=loader,
        session_service_uri=session_service_uri or "memory://",
        memory_service_uri=memory_service_uri,
        artifact_service_uri=artifact_service_uri,
        web=web,
        host=host,
        port=port,
    )
