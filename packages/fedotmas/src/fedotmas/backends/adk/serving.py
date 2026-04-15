from __future__ import annotations

from typing import Any, Union

from fastapi import FastAPI
from google.adk.agents.base_agent import BaseAgent
from google.adk.apps.app import App
from google.adk.cli.fast_api import get_fast_api_app
from google.adk.cli.utils.base_agent_loader import BaseAgentLoader
from google.adk.sessions.base_session_service import BaseSessionService

from fedotmas.backends.adk.builder import build_adk_tree
from fedotmas.backends.adk.middleware import MiddlewareAdapter
from fedotmas.interfaces.agent import AgentTree
from fedotmas.interfaces.middleware import MiddlewareProtocol


class _AgentLoader(BaseAgentLoader):
    """Agent loader that serves pre-built BaseAgent instances."""

    def __init__(self) -> None:
        self._agents: dict[str, BaseAgent | App] = {}

    def register(self, name: str, agent: BaseAgent | App) -> None:
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


def serve_adk(
    agent_trees: dict[str, AgentTree],
    *,
    middlewares: list[MiddlewareProtocol] | None = None,
    backend_plugins: list[Any] | None = None,
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
    """Create a FastAPI app that serves agent trees via ADK API server."""
    from google.adk.plugins import BasePlugin

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

    plugins: list[BasePlugin] = []
    if middlewares:
        plugins.append(MiddlewareAdapter(middlewares))
    if backend_plugins:
        plugins.extend(backend_plugins)

    loader = _AgentLoader()
    for name, tree in agent_trees.items():
        adk_agent = build_adk_tree(tree)
        app = App(name=name, root_agent=adk_agent, plugins=plugins)
        loader.register(name, app)

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
