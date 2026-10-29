"""Preserve provider truncation evidence even when structured output is invalid."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

from fedotmas.meta import _adk_runner as runner
from google.genai import types
from pydantic import BaseModel


class Config(BaseModel):
    name: str


def response(text, finish_reason="length"):
    return SimpleNamespace(
        finish_reason=finish_reason,
        content=types.Content(
            parts=[
                types.Part(text=text),
                types.Part(text="private reasoning", thought=True),
            ]
        ),
        usage_metadata=SimpleNamespace(
            prompt_token_count=10, candidates_token_count=8192
        ),
    )


async def test_truncation_feedback_survives_json_validation_error(
    model_config, monkeypatch, tmp_path
):
    monkeypatch.setenv("FEDOTMAS_META_RESPONSE_DIR", str(tmp_path / "responses"))
    messages = []
    raw = '{"name":"unfinished' + "\r" * 40

    async def execute(**kwargs):
        messages.append(kwargs["user_message"])
        tracker = runner._MetaUsageTracker(kwargs["usage_totals"])
        await tracker.after_model_callback(
            callback_context=None, llm_response=response(raw)
        )
        if len(messages) == 1:
            raise ValueError("Invalid JSON: EOF while parsing a string")
        return runner.LLMCallResult(
            raw_output={"name": "ok"},
            prompt_tokens=10,
            completion_tokens=8192,
            elapsed=0.1,
        )

    monkeypatch.setattr(runner, "_execute_meta_call", execute)
    monkeypatch.setattr(runner.asyncio, "sleep", AsyncMock())
    result = await runner.run_meta_agent_call(
        agent_name="test",
        instruction="generate",
        user_message="task",
        output_schema=Config,
        output_key="config",
        model=model_config,
        temperature=0.3,
        max_retries=1,
    )
    assert "truncated by the output-token limit" in messages[1]
    assert result.completion_tokens == 16384
    files = list((tmp_path / "responses").glob("*.json"))
    assert len(files) == 2
    for path in files:
        saved = json.loads(path.read_text())
        assert saved["text"] == raw
        assert saved["finish_reason"] == "length"
        assert saved["completion_tokens"] == 8192
        assert "private reasoning" not in path.read_text()
        assert path.stat().st_mode & 0o777 == 0o600


async def test_response_capture_is_opt_in(monkeypatch, tmp_path):
    monkeypatch.delenv("FEDOTMAS_META_RESPONSE_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    usage = {}
    await runner._MetaUsageTracker(usage).after_model_callback(
        callback_context=None, llm_response=response("{}", "stop")
    )
    assert list(tmp_path.iterdir()) == []
    assert usage["completion"] == 8192


async def test_unwritable_diagnostic_directory_does_not_fail_model_call(
    monkeypatch, tmp_path
):
    path = tmp_path / "file"
    path.write_text("not a directory")
    monkeypatch.setenv("FEDOTMAS_META_RESPONSE_DIR", str(path))
    usage = {}
    await runner._MetaUsageTracker(usage).after_model_callback(
        callback_context=None, llm_response=response("{}", "stop")
    )
    assert usage["completion"] == 8192
    assert path.read_text() == "not a directory"
