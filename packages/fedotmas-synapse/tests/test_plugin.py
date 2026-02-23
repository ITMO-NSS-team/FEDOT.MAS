from __future__ import annotations

from fedotmas_synapse.otel import OtelEventCallback
from fedotmas_synapse.plugin import SynapsePlugin
from fedotmas_synapse.session import MongoSessionService


def test_plugin_creates_services(plugin: SynapsePlugin) -> None:
    assert isinstance(plugin.session_service, MongoSessionService)
    assert isinstance(plugin.checkpoint, object)
    assert plugin.otel_callback is None


def test_plugin_with_otel(mongo_uri: str) -> None:
    p = SynapsePlugin(mongo_uri, otel_endpoint="http://localhost:4318/v1/traces")
    assert isinstance(p.otel_callback, OtelEventCallback)


def test_mas_kwargs_structure(plugin: SynapsePlugin) -> None:
    kw = plugin.mas_kwargs()
    assert "session_service" in kw
    assert "before_agent_callbacks" in kw
    assert "after_agent_callbacks" in kw
    assert isinstance(kw["before_agent_callbacks"], list)
    assert isinstance(kw["after_agent_callbacks"], list)
    assert all(callable(cb) for cb in kw["before_agent_callbacks"])
    assert all(callable(cb) for cb in kw["after_agent_callbacks"])
    assert "event_callback" not in kw


def test_mas_kwargs_with_otel(mongo_uri: str) -> None:
    p = SynapsePlugin(mongo_uri, otel_endpoint="http://localhost:4318/v1/traces")
    kw = p.mas_kwargs()
    assert "event_callback" in kw
    assert isinstance(kw["event_callback"], OtelEventCallback)
