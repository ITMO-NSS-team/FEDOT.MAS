from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from generate_sampo_phase_6 import render_prompt
from sampo_phase_6 import (
    LIMITS,
    TASK,
    BatchTrace,
    _find_tool_error,
    assert_neutral_manual_inputs,
    manual_input_audit_text,
)
from run_sampo_phase_6 import structural_check
from fedotmas.mas.models import MASAgentConfig, MASConfig


EXPECTED_TASK = (
    "Map each assigned public historical construction work name to three distinct allowed labels. "
    "Produce valid top-3 predictions for every assigned ID. Use only public data and the supplied tools. "
    "Process only the assigned IDs."
)


def test_phase6_task_is_exact_neutral_task() -> None:
    assert TASK == EXPECTED_TASK


def test_phase6_meta_prompt_has_only_routing_mechanics_and_no_examples() -> None:
    prompt = render_prompt("MCP SERVER: sampo-phase6\nTOOL: prepare_candidates", ["openai/gpt-5-mini"])
    assert "coordinator" in prompt
    assert "call-and-return" in prompt
    assert "MASConfig, a worker's tools list contains MCP server names" in prompt
    assert "AVAILABLE WORKER MODELS" in prompt
    assert "Example 1" not in prompt
    assert "Research + synthesis" not in prompt
    assert "disagreement" not in prompt.casefold()
    assert "uncertainty" not in prompt.casefold()
    assert "semantic review" not in prompt.casefold()
    assert "complementary retrieval" not in prompt.casefold()


def test_manual_input_audit_rejects_policy_language() -> None:
    payload = manual_input_audit_text(
        "neutral config prompt",
        EXPECTED_TASK,
        "Public tool server",
        [
            {
                "name": "prepare_candidates",
                "description": "Create candidate results for a public batch.",
                "inputSchema": {"type": "object"},
            }
        ],
        [{"purpose": "test", "message": "Batch offset: 0"}],
    )
    assert_neutral_manual_inputs(payload)
    payload["tools"][0]["description"] = "Use a disagreement gate before inspection."
    with pytest.raises(ValueError, match="forbidden policy language"):
        assert_neutral_manual_inputs(payload)


def test_phase6_surface_excludes_phase5_policy_tools_and_parameter() -> None:
    source_path = ROOT / "mcp-servers" / "sampo-phase6" / "src" / "mcp_sampo_phase6" / "server.py"
    source = source_path.read_text(encoding="utf-8")
    for name in (
        "partition_candidate_batch",
        "stage_candidate_predictions",
        "get_run_status",
        "finalize_predictions",
        "fill_retrieval_tail",
    ):
        assert name not in source


def test_structural_check_only_rejects_inaccessible_tools_and_models() -> None:
    config = MASConfig(
        coordinator=MASAgentConfig(
            name="coordinator",
            description="Controls execution.",
            instruction="Call a worker with the supplied task.",
            model="openai/gpt-5-mini",
        ),
        workers=[
            MASAgentConfig(
                name="worker",
                description="Handles the task.",
                instruction="Complete the supplied task.",
                model="openai/gpt-5-mini",
                tools=["sampo-phase6"],
            )
        ],
    )
    assert structural_check(config, {"openai/gpt-5-mini"}) == []
    invalid = config.model_copy(
        update={
            "workers": [
                config.workers[0].model_copy(
                    update={"tools": ["unknown-server"], "model": "other/model"}
                )
            ]
        }
    )
    failures = structural_check(invalid, {"openai/gpt-5-mini"})
    assert any(item.startswith("inaccessible_tools:") for item in failures)
    assert any(item.startswith("worker_model_not_available:") for item in failures)


def test_neutral_tool_description_does_not_encode_call_policy() -> None:
    source_path = ROOT / "mcp-servers" / "sampo-phase6" / "src" / "mcp_sampo_phase6" / "server.py"
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    tool_names = {
        "list_methods",
        "prepare_candidates",
        "inspect_candidates",
        "save_default_top3",
        "save_ranked_top3",
        "get_prediction_status",
    }
    tool_descriptions = [
        ast.get_docstring(node) or ""
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name in tool_names
    ]
    assert len(tool_descriptions) == 6
    all_text = "\n".join(tool_descriptions).casefold()
    for term in ("disagreement", "uncertainty", "fallback", "semantic reranking", "recommended call order"):
        assert term not in all_text


def test_model_call_budget_short_circuits_before_an_extra_provider_call() -> None:
    import asyncio
    from types import SimpleNamespace

    trace = BatchTrace(["10477"], 0, "phase6_test_budget")
    trace.model_request_attempts = LIMITS["max_model_calls"]
    response = asyncio.run(
        trace.before_model_callback(
            callback_context=None,
            llm_request=SimpleNamespace(model="openai/gpt-5-mini"),
        )
    )
    assert response is not None
    assert trace.failures == ["model_call_limit"]


def test_tool_call_budget_short_circuits_before_an_extra_tool_execution() -> None:
    import asyncio
    from types import SimpleNamespace

    trace = BatchTrace(["10477"], 0, "phase6_test_budget")
    trace.tool_calls = [{} for _ in range(LIMITS["max_tool_calls"])]
    response = asyncio.run(
        trace.before_tool_callback(
            tool=SimpleNamespace(name="list_methods"),
            tool_args={},
            tool_context=SimpleNamespace(function_call_id="call_over_limit"),
        )
    )
    assert response == {"error": "Phase 6 tool-call limit reached. Stop execution."}
    assert trace.failures == ["tool_call_limit"]


def test_mcp_error_result_records_only_the_conflicting_assigned_ids() -> None:
    from types import SimpleNamespace

    result = {
        "content": [
            {"type": "text", "text": "Prediction conflicts with an existing stored top-3 for IDs: 410"}
        ],
        "isError": True,
    }
    message = _find_tool_error(result)
    assert message and "IDs: 410" in message

    trace = BatchTrace(["410", "4507"], 0, "phase6_test_conflict")
    trace._record_tool_error(
        "save_ranked_top3",
        {"rankings": [{"example_id": "410"}, {"example_id": "4507"}]},
        message,
    )
    assert trace.conflicting_rewrites == ["410"]
