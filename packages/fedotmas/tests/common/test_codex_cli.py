from __future__ import annotations

import json
from pathlib import Path

import pytest
from fedotmas.common import codex_cli
from fedotmas.maw.models import AgentPoolConfig, MAWConfig
from fedotmas.mas.models import MASConfig


class _FakeProcess:
    returncode = 0

    async def communicate(self, value: bytes):
        assert value.decode("utf-8") == "hello"
        return (
            b'{"type":"turn.completed","usage":{"input_tokens":12,"output_tokens":5}}\n',
            b"",
        )


async def test_run_codex_cli_uses_subscription_safe_flags(monkeypatch, tmp_path):
    captured: list[str] = []

    async def fake_subprocess(*args, **kwargs):
        del kwargs
        captured.extend(str(arg) for arg in args)
        schema_path = Path(args[args.index("--output-schema") + 1])
        assert json.loads(schema_path.read_text(encoding="utf-8")) == {
            "type": "object", "properties": {},
            "required": [], "additionalProperties": False,
        }
        final_path = Path(args[args.index("--output-last-message") + 1])
        final_path.write_text("done", encoding="utf-8")
        return _FakeProcess()

    monkeypatch.setattr(codex_cli, "find_codex_cli", lambda: "codex")

    async def logged_in():
        return True, "Logged in using ChatGPT"

    monkeypatch.setattr(codex_cli, "codex_login_status", logged_in)
    monkeypatch.setattr(codex_cli.asyncio, "create_subprocess_exec", fake_subprocess)
    result = await codex_cli.run_codex_cli(
        "host/gpt-5.6-terra", "hello", workdir=tmp_path,
        output_schema={"type": "object", "properties": {}},
    )

    assert result.text == "done"
    assert result.prompt_tokens == 12
    assert result.completion_tokens == 5
    assert captured[:4] == ["codex", "-a", "never", "exec"]
    assert "--ignore-user-config" in captured
    assert "--ignore-rules" in captured
    assert "--ephemeral" in captured
    assert captured[captured.index("-m") + 1] == "gpt-5.6-terra"


def test_decision_part_converts_function_call():
    part = codex_cli._decision_part(
        json.dumps(
            {
                "kind": "function_call",
                "text": "",
                "function_name": "predict_rubber_properties",
                "arguments_json": '{"nr_smr20_phr":55}',
            }
        ),
        {"predict_rubber_properties"},
    )
    assert part.function_call.name == "predict_rubber_properties"
    assert part.function_call.args == {"nr_smr20_phr": 55}


def test_decision_part_rejects_unavailable_function():
    with pytest.raises(RuntimeError, match="unavailable function"):
        codex_cli._decision_part(
            json.dumps(
                {
                    "kind": "function_call",
                    "text": "",
                    "function_name": "invented_tool",
                    "arguments_json": "{}",
                }
            ),
            {"predict_rubber_properties"},
        )


@pytest.mark.parametrize("model", [AgentPoolConfig, MAWConfig, MASConfig])
def test_generation_schemas_are_strict_and_original_is_unchanged(model):
    original = model.model_json_schema()
    before = json.dumps(original)
    strict = codex_cli._strict_output_schema(original)

    def check(node):
        if isinstance(node, dict):
            assert "default" not in node
            if node.get("type") == "object":
                assert node["additionalProperties"] is False
                assert node["required"] == list(node["properties"])
            for value in node.values():
                check(value)
        elif isinstance(node, list):
            for value in node:
                check(value)

    check(strict)
    assert json.dumps(original) == before
    assert codex_cli._strict_output_schema(strict) == strict
    entry = next(strict["$defs"][name] for name in
                 ("AgentPoolEntry", "MAWAgentConfig", "MASAgentConfig")
                 if name in strict["$defs"])
    assert {"type": "null"} in entry["properties"]["model"]["anyOf"]
    if model is MAWConfig:
        assert strict["$defs"]["MAWStepConfig"]["properties"]["children"]["items"] == {
            "$ref": "#/$defs/MAWStepConfig"
        }


def test_strict_schema_does_not_silently_discard_dictionary_values():
    with pytest.raises(ValueError, match="dynamic additionalProperties"):
        codex_cli._strict_output_schema({
            "type": "object", "additionalProperties": {"type": "string"},
        })


def test_tool_decision_schema_stays_unchanged():
    schema = codex_cli._tool_decision_schema(["predict_rubber_properties"])
    assert codex_cli._strict_output_schema(schema) == schema
