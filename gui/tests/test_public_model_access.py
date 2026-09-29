"""Public requests cannot spend the server owner's CLI subscription."""

import importlib
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI, HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
app = importlib.import_module("server.app")
security = importlib.import_module("server.security")
from server.schemas import RunIn  # noqa: E402


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["host/test", "codex/test"])
async def test_public_native_models_rejected_before_completion(monkeypatch, model):
    monkeypatch.setattr(security, "PUBLIC_MODE", True)
    monkeypatch.setattr(security, "ACCESS_TOKEN", "test-token")
    complete = AsyncMock()
    monkeypatch.setattr(app, "complete", complete)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app.app), base_url="http://localhost"
    ) as client:
        response = await client.post(
            "/api/baseline",
            json={"query": "Answer", "model": model},
            headers={"x-access-token": "test-token"},
        )
    assert response.status_code == 403
    complete.assert_not_called()


@pytest.mark.asyncio
async def test_public_native_model_inside_config_rejected(monkeypatch):
    monkeypatch.setattr(security, "PUBLIC_MODE", True)
    monkeypatch.setattr(app, "DEFAULT_MODEL", "openrouter/test")
    with pytest.raises(HTTPException) as error:
        await app.run(
            RunIn(
                kind="maw",
                config={
                    "agents": [
                        {
                            "name": "worker",
                            "instruction": "Answer",
                            "output_key": "answer",
                            "model": "host/test",
                        }
                    ],
                    "pipeline": {"type": "agent", "agent_name": "worker"},
                },
                query="Answer",
                tools=[],
            )
        )
    assert error.value.status_code == 403


@pytest.mark.asyncio
async def test_missing_judge_model_uses_judge_default_for_key_guard(monkeypatch):
    monkeypatch.setattr(security, "PUBLIC_MODE", False)
    monkeypatch.setattr(security, "DEFAULT_MODEL", "openrouter/test")
    monkeypatch.setattr(security, "JUDGE_MODEL", "host/test")
    monkeypatch.setattr(security, "_key_available", lambda *_: False)
    server = FastAPI()
    security.install(server)

    @server.post("/api/judge")
    async def judge():
        return {"ok": True}

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server), base_url="http://localhost"
    ) as client:
        response = await client.post("/api/judge", json={})
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_public_judge_skips_native_fallback(monkeypatch):
    judge = importlib.import_module("server.judge")
    from server.schemas import JudgeIn

    monkeypatch.setattr(security, "PUBLIC_MODE", True)
    monkeypatch.setattr(judge, "JUDGE_FALLBACK", "host/test")
    ask = AsyncMock(return_value="")
    direct = AsyncMock(return_value="")
    monkeypatch.setattr(judge, "_ask_judge", ask)
    monkeypatch.setattr(judge, "_ask_judge_direct", direct)
    result = await judge._judge_impl(
        JudgeIn(
            query="Answer",
            system_answer="42",
            single_answer="43",
            model="openrouter/test",
        )
    )
    assert result["ok"] is False
    assert ask.await_count == 3
    assert direct.await_count == 1
    assert all(
        call.args[0] == "openrouter/test"
        for call in ask.await_args_list + direct.await_args_list
    )
