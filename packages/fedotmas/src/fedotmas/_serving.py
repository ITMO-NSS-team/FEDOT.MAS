from __future__ import annotations

from typing import Union

from fastapi import FastAPI
from google.adk.agents.base_agent import BaseAgent
from google.adk.apps.app import App
from google.adk.cli.fast_api import get_fast_api_app
from google.adk.cli.utils.base_agent_loader import BaseAgentLoader
from google.adk.sessions.base_session_service import BaseSessionService


class _AgentLoader(BaseAgentLoader):
    """Agent loader that serves pre-built BaseAgent instances.

    Used by `serve` to bridge MAS-built agents to ADK's serving infrastructure.
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


def serve(
    agents: dict[str, BaseAgent | App],
    *,
    session_service: BaseSessionService | None = None,
    session_service_uri: str | None = None,
    memory_service_uri: str | None = None,
    artifact_service_uri: str | None = None,
    web: bool = False,
    host: str = "127.0.0.1",
    port: int = 8000,
    allow_origins: list[str] | None = None,
    auto_create_session: bool = False,
) -> FastAPI:
    """Create a FastAPI app that serves MAS-generated agents via ADK API server.

    Args:
        agents: Mapping of agent name to BaseAgent or App instance.
        session_service: A BaseSessionService instance for backends not
            natively supported by ADK (e.g. MongoSessionService from
            fedotmas-synapse). Mutually exclusive with ``session_service_uri``.
        session_service_uri: URI for ADK built-in session backends:
            ``"memory://"``, ``"sqlite:///sessions.db"``,
            ``"postgresql://..."``, ``"mysql://..."``.
            Defaults to in-memory.
        memory_service_uri: URI for memory service. Defaults to None.
        artifact_service_uri: URI for artifact storage. Defaults to None.
        web: If True, include the ADK web UI.
        host: Bind address.
        port: Bind port.
        allow_origins: CORS allowed origins (e.g. ``["*"]`` or
            ``["https://example.com"]``).
        auto_create_session: If True, automatically create a session when
            ``/run`` or ``/run_sse`` is called without an existing one.
    """
    if session_service is not None and session_service_uri is not None:
        raise ValueError(
            "session_service and session_service_uri are mutually exclusive"
        )

    if session_service is not None:
        from google.adk.cli.service_registry import get_service_registry

        def _factory(uri: str, **kwargs: object) -> BaseSessionService:
            return session_service

        scheme = f"fedotmas-instance-{id(session_service)}"
        get_service_registry().register_session_service(scheme, _factory)
        session_service_uri = f"{scheme}://"

    loader = _AgentLoader()
    for name, agent in agents.items():
        loader.register(name, agent)

    return get_fast_api_app(
        agents_dir=".",
        agent_loader=loader,
        session_service_uri=session_service_uri or "memory://",
        memory_service_uri=memory_service_uri,
        artifact_service_uri=artifact_service_uri,
        use_local_storage=False,
        web=web,
        host=host,
        port=port,
        allow_origins=allow_origins,
        auto_create_session=auto_create_session,
    )
