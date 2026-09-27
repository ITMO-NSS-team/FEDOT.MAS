"""The GUI source download is a runnable bundle, not an LLM-generated guess."""

import ast
import importlib
import io
import json
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
        assert manifest == {"kind": kind, "tools": [], "custom_mcp": [],
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
