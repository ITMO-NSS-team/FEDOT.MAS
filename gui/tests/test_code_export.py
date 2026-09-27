"""The GUI source download is a runnable bundle, not an LLM-generated guess."""

import ast
import importlib
import io
import json
import runpy
import sys
import zipfile
from pathlib import Path

import pytest
from fedotmas import MASConfig, MAWConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
app = importlib.import_module("server.app")
from server.schemas import CodeExportIn, CustomMCP


def configuration(kind):
    agent = {"name": "сборщик", "instruction": "Реши задачу и вызови расчётчик.",
             "model": "openrouter/qwen/qwen3-32b", "tools": [], "output_key": "result"}
    if kind == "mas":
        return {"coordinator": dict(agent, name="координатор", description="Назначает задачи"),
                "workers": [dict(agent, name="расчётчик", description="Считает данные")]}
    return {"agents": [agent], "pipeline": {"type": "agent", "agent_name": "сборщик"}}


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["mas", "maw"])
async def test_archive_contains_valid_runnable_code_and_config(kind):
    source = configuration(kind)
    body = CodeExportIn(kind=kind, config=source, tools=[], model="openrouter/qwen/qwen3-32b")
    response = await app.export_code(body)
    assert response.media_type == "application/zip"
    assert response.headers["cache-control"] == "no-store"
    with zipfile.ZipFile(io.BytesIO(response.body)) as archive:
        assert set(archive.namelist()) == {"run.py", "config.json", "manifest.json", "README.md"}
        code = archive.read("run.py").decode()
        ast.parse(code)
        assert "build_and_run(config, query)" in code
        config = json.loads(archive.read("config.json"))
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest == {"kind": kind, "tools": [], "custom_mcp": [], "bundled_mcp": [],
                            "model": "openrouter/qwen/qwen3-32b"}
        (MASConfig if kind == "mas" else MAWConfig).model_validate(config)
        if kind == "mas":
            assert config["workers"][0]["name"] == "raschetchik"
            assert "raschetchik" in config["coordinator"]["instruction"]
        else:
            assert config["pipeline"]["agent_name"] == "сборщик"
    assert body.config == source


@pytest.mark.asyncio
async def test_custom_mcp_credentials_never_enter_archive():
    secret = "sk-or-v1-0123456789abcdef0123456789abcdef"
    source = configuration("maw")
    source["agents"][0]["tools"] = ["remote"]
    source["agents"][0]["instruction"] += " Ключ: " + secret
    body = CodeExportIn(config=source, kind="maw", tools=[],
                        custom_mcp=[CustomMCP(name="remote",
                          url="https://private.example.test/mcp?api_key=" + secret,
                          headers={"Authorization": "Bearer " + secret})])
    response = await app.export_code(body)
    with zipfile.ZipFile(io.BytesIO(response.body)) as archive:
        all_bytes = b"".join(archive.read(name) for name in archive.namelist())
        assert secret.encode() not in all_bytes
        assert b"private.example.test" not in all_bytes
        config = json.loads(archive.read("config.json"))
        manifest = json.loads(archive.read("manifest.json"))
        assert config["agents"][0]["tools"] == ["remote"]
        assert manifest["custom_mcp"] == ["remote"]


@pytest.mark.asyncio
async def test_invalid_config_returns_validation_error():
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as raised:
        await app.export_code(CodeExportIn(config={}, tools=[]))
    assert raised.value.status_code == 422


@pytest.mark.asyncio
@pytest.mark.parametrize("name,module", [
    ("rubber-recipe-predictor", "mcp_rubber_recipe_predictor"),
    ("technology-card-audit", "mcp_technology_card_audit"),
    ("sandbox-light", "mcp_sandbox_light"),
])
async def test_selected_calculation_mcp_is_bundled_with_sources(name, module):
    source = configuration("maw")
    source["agents"][0]["tools"] = [name]
    response = await app.export_code(CodeExportIn(kind="maw", config=source, tools=[name]))
    with zipfile.ZipFile(io.BytesIO(response.body)) as archive:
        paths = set(archive.namelist())
        root = f"mcp-servers/{name}"
        assert f"{root}/pyproject.toml" in paths
        assert f"{root}/src/{module}/server.py" in paths
        assert f"{root}/src/{module}/__init__.py" in paths
        assert json.loads(archive.read("manifest.json"))["bundled_mcp"] == [name]
        assert "uv" in archive.read("README.md").decode()
        if name == "rubber-recipe-predictor":
            assert "experiments/rubber_recipe_mas/open_data_predictor.py" in paths
            assert "experiments/rubber_recipe_mas/open_data/tire_tread_sbr_nr.csv" in paths
            assert "experiments/rubber_recipe_mas/open_data/README.md" in paths
            assert len(archive.read("experiments/rubber_recipe_mas/open_data/tire_tread_sbr_nr.csv")) > 100
        assert not any("source/" in item or "_run/" in item for item in paths)
        assert "mcp-servers/rubber-recipe-predictor/pyproject.toml" not in paths or name == "rubber-recipe-predictor"


@pytest.mark.asyncio
async def test_runner_prefers_bundled_calculator_without_running_mas(tmp_path, monkeypatch, capsys):
    import fedotmas
    import fedotmas.mcp

    source = configuration("maw")
    source["agents"][0]["tools"] = ["rubber-recipe-predictor"]
    response = await app.export_code(CodeExportIn(
        kind="maw", config=source, tools=["rubber-recipe-predictor"]))
    with zipfile.ZipFile(io.BytesIO(response.body)) as archive:
        archive.extractall(tmp_path)

    predictor_path = tmp_path / "experiments" / "rubber_recipe_mas" / "open_data_predictor.py"
    spec = importlib.util.spec_from_file_location("exported_predictor", predictor_path)
    predictor = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "exported_predictor", predictor)
    spec.loader.exec_module(predictor)
    assert len(predictor.load_rows()) == 20

    seen = {}

    class FakeSystem:
        def __init__(self, **kwargs):
            seen["servers"] = kwargs["mcp_servers"]

        async def build_and_run(self, config, query):
            seen["query"] = query
            return {"answer": "ok"}

    def resolve(names):
        seen["resolved"] = names
        return {}

    monkeypatch.setattr(fedotmas, "MAW", FakeSystem)
    monkeypatch.setattr(fedotmas.mcp, "resolve_mcp_registry", resolve)
    monkeypatch.setattr(sys, "argv", ["run.py", "Проверка"])
    exported = runpy.run_path(str(tmp_path / "run.py"), run_name="source_test")
    await exported["main"]()
    assert seen["resolved"] == []
    assert seen["query"] == "Проверка"
    server = seen["servers"]["rubber-recipe-predictor"]
    assert str(tmp_path / "mcp-servers" / "rubber-recipe-predictor") in server.args
    assert "mcp_rubber_recipe_predictor.server" in server.args
    assert '"answer": "ok"' in capsys.readouterr().out
