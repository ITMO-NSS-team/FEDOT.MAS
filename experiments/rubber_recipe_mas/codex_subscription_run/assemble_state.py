from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL = "host/gpt-5.6-sol"

OUTPUT_FILES = {
    "data_contract_output": "data_contract_output.md",
    "chemistry_output": "chemistry_output.md",
    "ml_validation_output": "ml_validation_output.md",
    "final_report": "final_report.md",
}


def main() -> None:
    routing = json.loads((HERE / "001_routing.json").read_text(encoding="utf-8"))
    config = json.loads((HERE.parent / "config.json").read_text(encoding="utf-8"))
    task = (HERE.parent / "task.md").read_text(encoding="utf-8")
    state = {
        "user_query": task,
        **{
            key: (HERE / filename).read_text(encoding="utf-8")
            for key, filename in OUTPUT_FILES.items()
        },
    }

    call_sequence = [
        {
            "sequence": 1,
            "agent_name": "rubber_recipe_master",
            "kind": "routing",
            "model": MODEL,
            "output": "001_routing.json",
        },
        *[
            {
                "sequence": index,
                "agent_name": call["agent_name"],
                "kind": "worker",
                "model": MODEL,
                "request": call["request"],
                "output_key": output_key,
                "output": OUTPUT_FILES[output_key],
            }
            for index, (call, output_key) in enumerate(
                zip(
                    routing["calls"],
                    (
                        "data_contract_output",
                        "chemistry_output",
                        "ml_validation_output",
                    ),
                    strict=True,
                ),
                start=2,
            )
        ],
        {
            "sequence": 5,
            "agent_name": "rubber_recipe_master",
            "kind": "synthesis",
            "model": MODEL,
            "inputs": list(OUTPUT_FILES)[:-1],
            "output_key": "final_report",
            "output": "final_report.md",
        },
    ]

    manifest = {
        "status": "completed",
        "workflow": "fedotmas.MAS master-orchestrator",
        "transport": "Codex host-native subscription",
        "native_litellm_api_run": False,
        "model": MODEL,
        "reasoning_effort": "xhigh",
        "call_count": len(call_sequence),
        "latency_seconds": None,
        "latency_note": (
            "The host subagent interface did not expose per-call latency; "
            "no value was inferred from filesystem timestamps."
        ),
        "provider_billing_tokens": None,
        "character_counts": {
            key: len(value) for key, value in state.items() if key != "user_query"
        },
        "call_sequence": call_sequence,
    }

    (HERE / "final_state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (HERE / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    expected_agents = [worker["name"] for worker in config["workers"]]
    routed_agents = [call["agent_name"] for call in routing["calls"]]
    expected_output_keys = {worker["output_key"] for worker in config["workers"]} | {
        config["coordinator"]["output_key"]
    }
    report = state["final_report"]
    coverage_terms = [
        "data contract",
        "leakage",
        "RDKit",
        "Morgan",
        "XGBoost",
        "каскад",
        "0,03",
        "SHAP",
        "RAG",
    ]
    checks = {
        "routing_model_is_exact": routing["model"] == MODEL,
        "configured_models_are_exact": all(
            agent["model"] == MODEL
            for agent in [config["coordinator"], *config["workers"]]
        ),
        "routing_uses_each_configured_worker_once": routed_agents == expected_agents,
        "state_keys_match_config": set(state) - {"user_query"} == expected_output_keys,
        "all_outputs_nonempty": all(state[key].strip() for key in expected_output_keys),
        "final_report_covers_required_topics": all(
            term.casefold() in report.casefold() for term in coverage_terms
        )
        and (
            "go/no-go" in report.casefold()
            or "контрольные решения" in report.casefold()
        ),
        "final_report_has_no_monthly_plan": "месяц" not in report.casefold(),
        "call_budget_respected": len(call_sequence) == 5,
    }
    verification = {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "notes": [
            "Structural and lexical checks do not replace review by a rubber technologist.",
            "Subscription-host execution is not a LiteLLM/API execution.",
        ],
    }
    (HERE / "verification.json").write_text(
        json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if verification["status"] != "pass":
        raise SystemExit("Verification failed; inspect verification.json")


if __name__ == "__main__":
    main()
