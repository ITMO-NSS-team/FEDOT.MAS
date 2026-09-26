"""GUI tool selection must match the registry used to build the system."""
import asyncio
import importlib
import sys
from pathlib import Path

import pytest
from fedotmas import MASConfig, MAWConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
app = importlib.import_module("server.app")
from server.schemas import GenerateIn, RunIn


def config_for(kind, tools):
    agent = dict(name="worker", instruction="Process the supplied input.",
                 model="host/gpt-5.6-terra", tools=tools, output_key="answer")
    if kind == "maw":
        return MAWConfig(agents=[agent], pipeline={"type": "agent", "agent_name": "worker"})
    return MASConfig(coordinator=dict(agent, name="coordinator", description="Route"),
                     workers=[dict(agent, description="Process input")])


def agents(config):
    return config.agents if isinstance(config, MAWConfig) else [config.coordinator, *config.workers]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["mas", "maw"])
@pytest.mark.parametrize("selected", [[], ["sandbox-light"]])
async def test_generation_only_assigns_selected_tools(monkeypatch, kind, selected):
    class FakeSystem:
        def __init__(self, **kwargs):
            assert set(kwargs["mcp_servers"]) == set(selected)

        async def generate_config(self, task):
            return config_for(kind, [])

    monkeypatch.setattr(app, "MAS" if kind == "mas" else "MAW", FakeSystem)
    result = await app._generate_impl(GenerateIn(
        task="Process input", kind=kind, tools=selected, web=False,
    ), asyncio.Queue())
    assert result["ok"]
    config = (MASConfig if kind == "mas" else MAWConfig)(**result["config"])
    assert all(set(a.tools) <= set(selected) for a in agents(config))
    if selected:
        assert any("sandbox-light" in a.tools for a in agents(config))
    else:
        assert all("Все арифметические действия выполняй в песочнице" not in a.instruction
                   for a in agents(config))


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["mas", "maw"])
async def test_saved_scenario_with_unselected_sandbox_can_build(monkeypatch, kind):
    built = []

    class FakeSystem:
        last_result = None

        def __init__(self, **kwargs):
            assert kwargs["mcp_servers"] == {}

        async def build_and_run(self, config, query):
            assert all(a.tools == [] for a in agents(config))
            assert all("не подключены" in a.instruction for a in agents(config))
            # Exercise the real builder with the same empty registry.
            if kind == "mas":
                from fedotmas.mas.builder import build_routing_system
                build_routing_system(config, mcp_registry={})
            built.append(config)
            return {"answer": "ok"}

    monkeypatch.setattr(app, "MAS" if kind == "mas" else "MAW", FakeSystem)
    response = await app.run(RunIn(kind=kind, tools=[], query="test",
                                  config=config_for(kind, ["sandbox-light"]).model_dump()))
    events = [chunk async for chunk in response.body_iterator]
    assert built
    assert not any('"type": "error"' in str(chunk) for chunk in events)


def test_custom_tools_are_preserved_only_when_connected():
    from server.normalize import sanitize_config
    config = config_for("mas", ["custom_calculator", "sandbox-light"])
    sanitize_config(config, "mas", ["custom_calculator"],
                    available_tools={"custom_calculator": object()})
    assert all(a.tools == ["custom_calculator"] for a in agents(config))


@pytest.mark.asyncio
async def test_model_catalog_without_keys_lists_openrouter_before_codex(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    async def logged_out():
        return False, "Not logged in"

    monkeypatch.setattr(app, "codex_login_status", logged_out)
    result = await app.status()
    ids = [m["id"] for m in result["models"]]
    assert not result["openrouter_ready"]
    assert len([m for m in ids if m.startswith("openrouter/")]) == 4
    assert len([m for m in ids if m.startswith("host/")]) == 3
    assert all(m.startswith("openrouter/") for m in ids[:4])
    assert all(m.startswith("host/") for m in ids[4:])
