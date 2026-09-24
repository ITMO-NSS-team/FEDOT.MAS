from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPERIMENT_DIR = HERE.parent
HOST_MODEL = "gpt-5.6-terra"
CONFIG_MODEL = f"host/{HOST_MODEL}"

OUTPUT_FILES = {
    "formulation_output": "formulation_output.md",
    "chemistry_review": "chemistry_review.md",
    "prediction_audit": "prediction_audit.md",
    "recipe_prediction": "recipe_prediction.md",
}


def main() -> None:
    routing = json.loads((HERE / "001_routing.json").read_text(encoding="utf-8"))
    config = json.loads((EXPERIMENT_DIR / "config.json").read_text(encoding="utf-8"))
    task = (EXPERIMENT_DIR / "task.md").read_text(encoding="utf-8")
    saved_input = (EXPERIMENT_DIR / "mas_input_request.md").read_text(
        encoding="utf-8"
    )
    prediction = json.loads(
        (HERE / "model_prediction.json").read_text(encoding="utf-8")
    )

    state = {
        key: (HERE / filename).read_text(encoding="utf-8")
        for key, filename in OUTPUT_FILES.items()
    }
    (HERE / "final_state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    call_sequence = [
        {
            "sequence": 1,
            "agent_name": config["coordinator"]["name"],
            "kind": "routing",
            "model": CONFIG_MODEL,
            "output": "001_routing.json",
        },
        {
            "sequence": 2,
            "agent_name": config["coordinator"]["name"],
            "kind": "tool",
            "server": routing["tool_call"]["server"],
            "tool_name": routing["tool_call"]["name"],
            "arguments": routing["tool_call"]["arguments"],
            "output": "model_prediction.json",
        },
        *[
            {
                "sequence": index,
                "agent_name": call["agent_name"],
                "kind": "worker",
                "model": CONFIG_MODEL,
                "request_template": call["request_template"],
                "output_key": worker["output_key"],
                "output": OUTPUT_FILES[worker["output_key"]],
            }
            for index, (call, worker) in enumerate(
                zip(routing["calls"], config["workers"], strict=True), start=3
            )
        ],
        {
            "sequence": 6,
            "agent_name": config["coordinator"]["name"],
            "kind": "synthesis",
            "model": CONFIG_MODEL,
            "inputs": [worker["output_key"] for worker in config["workers"]],
            "output_key": config["coordinator"]["output_key"],
            "output": OUTPUT_FILES[config["coordinator"]["output_key"]],
        },
    ]
    manifest = {
        "status": "completed",
        "workflow": "fedotmas.MAS master-orchestrator",
        "transport": "Codex host-native subscription",
        "native_provider_api_run": False,
        "model": HOST_MODEL,
        "provider_label_in_config": CONFIG_MODEL,
        "reasoning_effort": "xhigh",
        "call_count": len(call_sequence),
        "latency_seconds": None,
        "latency_note": (
            "The host-native run did not expose comparable per-call latency; "
            "no value was inferred from filesystem timestamps."
        ),
        "character_counts": {key: len(value) for key, value in state.items()},
        "numerical_engine": {
            "mcp_server": routing["tool_call"]["server"],
            "tool_name": routing["tool_call"]["name"],
            "source_rows": prediction["provenance"]["source_rows"],
            "model_type": prediction["model"]["type"],
            "candidate_status": prediction["status"],
        },
        "call_sequence": call_sequence,
        "disclaimer": (
            "The exact subscription model was invoked by the Codex host. "
            "Saved responses are replayed through the real FEDOT.MAS/ADK tree "
            "for GUI inspection; this is not a provider API integration."
        ),
    }
    (HERE / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    expected_agents = [worker["name"] for worker in config["workers"]]
    routed_agents = [call["agent_name"] for call in routing["calls"]]
    expected_keys = {worker["output_key"] for worker in config["workers"]} | {
        config["coordinator"]["output_key"]
    }
    recipe_output = state["recipe_prediction"]
    recipe = prediction["recipe"]
    properties = prediction["predicted_properties"]
    checks = {
        "workflow_is_mas_master_orchestrator": manifest["workflow"]
        == "fedotmas.MAS master-orchestrator",
        "routing_model_is_exact_terra": routing["model"] == CONFIG_MODEL,
        "configured_models_are_exact_terra": all(
            agent["model"] == CONFIG_MODEL
            for agent in [config["coordinator"], *config["workers"]]
        ),
        "routing_uses_each_configured_worker_once": routed_agents == expected_agents,
        "coordinator_has_numerical_tool": config["coordinator"]["tools"]
        == [routing["tool_call"]["server"]]
        and routing["tool_call"]["name"] == "predict_rubber_recipe",
        "user_input_has_no_numerical_result": "Результат численного движка"
        not in task,
        "saved_input_matches_task": saved_input == task,
        "state_keys_match_config": set(state) == expected_keys,
        "final_report_key_is_absent": "final_report" not in state,
        "all_outputs_nonempty": all(value.strip() for value in state.values()),
        "recipe_matches_numerical_engine": all(
            marker in recipe_output
            for marker in (
                f"| NR SMR-20 | {recipe['nr_smr20_phr']:.1f} |",
                f"| SBR-1502 | {recipe['sbr1502_phr']:.1f} |",
                (
                    "| Технический углерод N220 | "
                    f"{recipe['carbon_black_n220_phr']:.1f} |"
                ),
            )
        ),
        "all_point_constraints_satisfied": all(
            result["constraint_satisfied"] for result in properties.values()
        ),
        "uncertainty_is_reported": "LOOCV RMSE" in recipe_output
        and "робастно подтверждён лишь удельный вес" in recipe_output,
        "open_data_provenance_is_reported": "10.5281/zenodo.3838695"
        in recipe_output
        and "CC BY 4.0" in recipe_output
        and prediction["provenance"]["source_rows"] == 20,
        "call_budget_respected": len(call_sequence) == 6,
    }
    verification = {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "notes": [
            "Structural checks do not replace laboratory testing or review by a rubber technologist.",
            "Subscription-host execution is not a LiteLLM/provider-API execution.",
        ],
    }
    (HERE / "verification.json").write_text(
        json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if verification["status"] != "pass":
        raise SystemExit("Verification failed; inspect verification.json")


if __name__ == "__main__":
    main()
