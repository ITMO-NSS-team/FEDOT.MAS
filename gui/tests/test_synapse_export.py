"""GUI export uses the same converter as the supplied standalone script."""
import importlib
import sys
from pathlib import Path

import pytest
from fedotmas import MAWConfig
from fedotmas.export import to_synapse_bundle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
app = importlib.import_module("server.app")
from server.schemas import SynapseExportIn


def config():
    return dict(agents=[dict(name="analyst", instruction="Analyse the input",
        model="test/model", output_key="answer", tools=["rubber.predict"])],
        pipeline=dict(type="agent", agent_name="analyst"))


@pytest.mark.asyncio
async def test_export_matches_script():
    original = config()
    body = SynapseExportIn(config=original, workflow_id="tires", workflow_name="Шины")
    result = await app.export_synapse(body)
    expected = to_synapse_bundle(MAWConfig.model_validate(original),
                                workflow_id="tires", workflow_name="Шины")
    assert result["ok"]
    assert result["bundle"] == expected.bundle
    assert result["tools_checked"] is False
    assert body.config == original
    assert result["linearized_branches"] == 0
    assert result["degraded_loops"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("kind,cfg", [("mas", config()), ("maw", {}),
                                      ("maw", {"coordinator": {}})])
async def test_invalid_or_unsupported_config(kind, cfg):
    result = await app.export_synapse(SynapseExportIn(
        config=cfg, kind=kind, workflow_id="demo", workflow_name="Demo"))
    assert result["ok"] is False
    assert result["error"]
    assert "bundle" not in result


@pytest.mark.asyncio
async def test_topology_warnings():
    cfg = config()
    cfg["agents"].append(dict(cfg["agents"][0], name="second", output_key="second_result"))
    cfg["pipeline"] = {"type": "parallel", "children": [cfg["pipeline"],
        {"type": "agent", "agent_name": "second"}]}
    result = await app.export_synapse(SynapseExportIn(
        config=cfg, workflow_id="demo", workflow_name="Demo"))
    assert result["ok"]
    assert result["linearized_branches"] > 0


@pytest.mark.asyncio
async def test_mas_converted_to_sequential_maw():
    agent = dict(name="worker", description="Worker", instruction="Process input",
                 model="test/model", tools=["rubber.predict"], output_key=None)
    cfg = dict(coordinator=dict(agent, name="coordinator", output_key="export_result_1"),
               workers=[agent])
    result = await app.export_synapse(SynapseExportIn(
        config=cfg, kind="mas", workflow_id="demo", workflow_name="Demo"))
    assert result["ok"], result
    items = result["bundle"]["items"]
    assert len(items["agents"]) == 2
    assert len({a["output_save_key"] for a in items["agents"]}) == 2
    assert items["agents"][1]["system_prompt"] == agent["instruction"]
    phases = [n for n in items["workflows"][0]["nodes"] if n["type"] == "phase"]
    assert [n["agent_type"] for n in phases] == [a["_id"] for a in items["agents"]]
    assert cfg["workers"][0]["output_key"] is None
