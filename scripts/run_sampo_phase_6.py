"""Policy-neutral Phase 6 qualification and 3x20 comparison smoke."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "mcp-servers" / "sampo-phase6" / "src"))

from fedotmas.mas.models import MASConfig
from sampo_baselines import tfidf_word_ranked
from sampo_phase_6 import (
    BATCH_SIZE,
    LIMITS,
    OUT_ROOT,
    PREDICTIONS,
    SERVER_NAMES,
    WORKER_MODEL,
    assigned_ids,
    assert_neutral_manual_inputs,
    atomic_json,
    compose_task,
    labels,
    pilot_rows,
    prediction_path,
    read_stored,
    run_generated_batch,
    run_single_agent_batch,
)


def _latest_batch() -> Path:
    batches = sorted(path for path in OUT_ROOT.glob("batch_*") if path.is_dir())
    if not batches:
        raise FileNotFoundError("No generated Phase 6 batch exists")
    return batches[-1]


def structural_check(config: MASConfig, available_models: set[str]) -> list[str]:
    errors: list[str] = []
    if config.coordinator.model not in available_models:
        errors.append("coordinator_model_not_available")
    inaccessible_coordinator_tools = set(config.coordinator.tools) - SERVER_NAMES
    if inaccessible_coordinator_tools:
        errors.append(
            "inaccessible_tools:coordinator:" + ",".join(sorted(inaccessible_coordinator_tools))
        )
    if not config.workers:
        errors.append("missing_workers")
    for agent in config.workers:
        if agent.model not in available_models:
            errors.append(f"worker_model_not_available:{agent.name}")
        invalid = set(agent.tools) - SERVER_NAMES
        if invalid:
            errors.append(f"inaccessible_tools:{agent.name}:{','.join(sorted(invalid))}")
    return errors


def deterministic_baseline(offset: int, limit: int) -> dict[str, Any]:
    examples = pilot_rows()[offset : offset + limit]
    target_labels = labels()
    rankings = tfidf_word_ranked(
        [row["raw_work_name"] for row in examples], target_labels, 3
    )
    output = [
        {
            "example_id": row["example_id"],
            "top_1": ranking[0][0],
            "top_2": ranking[1][0],
            "top_3": ranking[2][0],
        }
        for row, ranking in zip(examples, rankings, strict=True)
    ]
    return {
        "offset": offset,
        "assigned_ids": [row["example_id"] for row in examples],
        "predictions": output,
        "method": "word_tfidf",
        "model_calls": 0,
        "tool_calls": 0,
    }


def refresh_run_prefix(batch_dir: Path) -> str:
    """Archive an aborted infrastructure attempt and refresh scoped run IDs."""
    audit_path = batch_dir / "phase6_leakage_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    old_prefix = audit["run_prefix"]
    new_prefix = f"phase6_retry_{uuid.uuid4().hex[:12]}"
    if list(PREDICTIONS.glob(f"{new_prefix}*.jsonl")):
        raise FileExistsError("Generated Phase 6 run prefix already exists")
    archive_index = 1
    while (batch_dir / f"qualification_aborted_attempt_{archive_index:02d}.json").exists():
        archive_index += 1
    archive_path = batch_dir / f"qualification_aborted_attempt_{archive_index:02d}.json"
    previous_qualification = batch_dir / "qualification_report.json"
    if previous_qualification.exists():
        previous = json.loads(previous_qualification.read_text(encoding="utf-8"))
        if previous.get("selected_config_index") is not None:
            raise FileExistsError("A config was already selected; refusing to refresh run IDs")
        report_index = 1
        while (batch_dir / f"qualification_report_attempt_{report_index:02d}.json").exists():
            report_index += 1
        os.replace(
            previous_qualification,
            batch_dir / f"qualification_report_attempt_{report_index:02d}.json",
        )
    prior_runs = []
    for index in range(1, 6):
        run_id = f"{old_prefix}_qual_{index:02d}"
        rows = read_stored(run_id)
        prior_runs.append(
            {
                "run_id": run_id,
                "stored_count": len(rows),
                "stored_ids": [row.get("example_id") for row in rows],
            }
        )
    atomic_json(
        archive_path,
        {
            "status": "aborted_due_to_budget_gate_enforcement_update",
            "qualification_selection_used": False,
            "reason": "The ADK callback logged a post-call model budget exception but continued execution; Phase 6 call limits now short-circuit before exceeding the budget.",
            "runs": prior_runs,
        },
    )
    audit["run_prefix"] = new_prefix
    for item in audit["manual_inputs"]["harness_added_instructions"]:
        purpose = item["purpose"]
        if purpose.startswith("behavioral qualification config "):
            index = int(purpose.rsplit(" ", 1)[1])
            run_id, offset, ids = f"{new_prefix}_qual_{index:02d}", 0, assigned_ids(0, 5)
        else:
            system, offset_text = purpose.split(" smoke offset ", 1)
            offset = int(offset_text)
            run_id = f"{new_prefix}_{system}_{offset:04d}"
            ids = assigned_ids(offset, BATCH_SIZE)
        item["message"] = compose_task(run_id, ids, offset)
    audit["execution_limits"] = LIMITS
    audit.setdefault("superseded_run_prefixes", []).append(
        {
            "prefix": old_prefix,
            "reason": "aborted after confirming post-call budget callback errors did not stop ADK execution",
        }
    )
    assert_neutral_manual_inputs(audit["manual_inputs"])
    atomic_json(audit_path, audit)
    generation_report_path = batch_dir / "generation_report.json"
    generation_report = json.loads(generation_report_path.read_text(encoding="utf-8"))
    generation_report["run_prefix"] = new_prefix
    generation_report["superseded_run_prefixes"] = audit["superseded_run_prefixes"]
    generation_report["execution_limits"] = LIMITS
    atomic_json(generation_report_path, generation_report)
    return new_prefix


async def qualify(batch_dir: Path) -> dict[str, Any]:
    report_path = batch_dir / "qualification_report.json"
    if report_path.exists():
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        trace_adapter_failure = bool(previous.get("configs")) and all(
            "inaccessible_tool_name" in item.get("execution", {}).get("failures", [])
            and not item.get("execution", {}).get("stored_ids")
            for item in previous["configs"]
        )
        if not trace_adapter_failure:
            raise FileExistsError(f"Refusing to rerun qualification: {report_path}")
        for item in previous["configs"]:
            run_id = item.get("execution", {}).get("run_id")
            if run_id and read_stored(run_id):
                raise RuntimeError("Cannot repeat failed qualification after a durable write")
        archived = batch_dir / "qualification_report_attempt_01.json"
        if archived.exists():
            raise FileExistsError(f"Refusing to overwrite {archived}")
        os.replace(report_path, archived)
    audit_path = batch_dir / "phase6_leakage_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not audit.get("manual_input_neutrality_asserted"):
        raise RuntimeError("Phase 6 leakage audit is missing its neutrality assertion")

    configs_dir = batch_dir
    config_count = len(audit["generated_mas_configs"])
    if config_count != 5:
        raise RuntimeError(f"Expected exactly five generated configs, got {config_count}")
    available_models = set(audit["worker_models"])
    prior_reports = sorted(batch_dir.glob("qualification_report_attempt_*.json"))
    prior_by_index: dict[int, dict[str, Any]] = {}
    if prior_reports:
        prior_report = json.loads(prior_reports[-1].read_text(encoding="utf-8"))
        if prior_report.get("selected_config_index") is None:
            prior_by_index = {
                item["config_index"]: item for item in prior_report.get("configs", [])
            }
    ids = assigned_ids(0, 5)
    run_prefix = json.loads((batch_dir / "generation_report.json").read_text())["run_prefix"]
    results: list[dict[str, Any]] = []
    selected: int | None = None
    for index in range(1, config_count + 1):
        config_path = configs_dir / f"config_{index:02d}" / "config.json"
        config = MASConfig.model_validate_json(config_path.read_text(encoding="utf-8"))
        structural_errors = structural_check(config, available_models)
        item: dict[str, Any] = {
            "config_index": index,
            "structural_errors": structural_errors,
            "structural_pass": not structural_errors,
            "execution": None,
        }
        previous = prior_by_index.get(index)
        previous_execution = (previous or {}).get("execution") or {}
        previous_failures = set(previous_execution.get("failures", []))
        if (
            previous
            and previous.get("structural_pass")
            and previous_failures
            and "total_token_limit" not in previous_failures
            and previous_execution.get("max_prompt_tokens", 0)
            <= LIMITS["max_prompt_tokens_per_call"]
            and previous_execution.get("model_calls", 0) <= LIMITS["max_model_calls"]
            and previous_execution.get("tool_calls", 0) <= LIMITS["max_tool_calls"]
            and previous_execution.get("runtime_seconds", 0)
            <= LIMITS["timeout_seconds"] + 1
        ):
            item["execution"] = previous_execution
            item["execution_reused_from"] = prior_reports[-1].name
            item["behavioral_pass"] = False
            item["passed"] = False
            item["structural_pass"] = previous["structural_pass"]
            results.append(item)
            continue
        if not structural_errors:
            run_id = f"{run_prefix}_qual_{index:02d}"
            if prediction_path(run_id).exists():
                raise FileExistsError(f"Refusing to overwrite run ID {run_id}")
            execution = await run_generated_batch(config, ids, 0, run_id)
            item["execution"] = execution
            item["behavioral_pass"] = execution["passed"]
            item["passed"] = execution["passed"]
        else:
            item["behavioral_pass"] = False
            item["passed"] = False
        results.append(item)
        if item["passed"]:
            if selected is None:
                selected = index
    report = {
        "attempt": 2 if (batch_dir / "qualification_report_attempt_01.json").exists() else 1,
        "qualification_ids": ids,
        "selection_rule": "first config passing policy-neutral structural and behavioral checks",
        "reused_prior_checks": any(item.get("execution_reused_from") for item in results),
        "selected_config_index": selected,
        "configs": results,
        "accuracy_used": False,
        "private_ground_truth_used": False,
        "limits": LIMITS,
    }
    atomic_json(report_path, report)
    if selected is None:
        raise RuntimeError("No generated Phase 6 config passed policy-neutral qualification")
    return report


async def smoke(batch_dir: Path, selected: int) -> dict[str, Any]:
    report_path = batch_dir / "smoke_report.json"
    if report_path.exists():
        raise FileExistsError(f"Refusing to rerun smoke: {report_path}")
    audit = json.loads((batch_dir / "phase6_leakage_audit.json").read_text(encoding="utf-8"))
    run_prefix = json.loads((batch_dir / "generation_report.json").read_text())["run_prefix"]
    config = MASConfig.model_validate_json(
        (batch_dir / f"config_{selected:02d}" / "config.json").read_text(encoding="utf-8")
    )
    comparisons: dict[str, Any] = {
        "deterministic_tfidf": [],
        "single_agent": [],
        "fedotmas_generated": [],
    }
    stopped_after_incomplete_generated_batch = False
    planned_offsets = (0, 20, 40)
    for offset in planned_offsets:
        ids = assigned_ids(offset, BATCH_SIZE)
        comparisons["deterministic_tfidf"].append(deterministic_baseline(offset, BATCH_SIZE))
        single_id = f"{run_prefix}_single_{offset:04d}"
        mas_id = f"{run_prefix}_mas_{offset:04d}"
        if prediction_path(single_id).exists() or prediction_path(mas_id).exists():
            raise FileExistsError("Refusing to overwrite Phase 6 smoke prediction run")
        comparisons["single_agent"].append(
            await run_single_agent_batch(ids, offset, single_id)
        )
        generated_result = await run_generated_batch(config, ids, offset, mas_id)
        comparisons["fedotmas_generated"].append(generated_result)
        if not generated_result["passed"]:
            stopped_after_incomplete_generated_batch = True
            break

    report = {
        "selected_config_index": selected,
        "selected_architecture": audit["generated_mas_configs"][selected - 1],
        "worker_model": WORKER_MODEL,
        "planned_batch_offsets": list(planned_offsets),
        "batch_offsets": [item["offset"] for item in comparisons["deterministic_tfidf"]],
        "batch_size": BATCH_SIZE,
        "stopped_after_incomplete_generated_batch": stopped_after_incomplete_generated_batch,
        "stop_reason": (
            "generated_mas_incomplete_assigned_batch"
            if stopped_after_incomplete_generated_batch
            else None
        ),
        "systems": comparisons,
        "phase5_reference": {
            "status": "not_run_in_phase6_smoke",
            "runner": "scripts/run_sampo_phase_5_smoke.py",
            "role": "diagnostic_reference_only",
        },
        "private_ground_truth_used": False,
        "accuracy_used": False,
        "limits": LIMITS,
    }
    atomic_json(report_path, report)
    return report


async def single_only_smoke(batch_dir: Path) -> dict[str, Any]:
    audit = json.loads(
        (batch_dir / "phase6_leakage_audit.json").read_text(encoding="utf-8")
    )
    run_prefix = json.loads(
        (batch_dir / "generation_report.json").read_text(encoding="utf-8")
    )["run_prefix"]
    systems: dict[str, Any] = {"deterministic_tfidf": [], "single_agent": []}
    for offset in (0, 20, 40):
        ids = assigned_ids(offset, BATCH_SIZE)
        systems["deterministic_tfidf"].append(deterministic_baseline(offset, BATCH_SIZE))
        run_id = f"{run_prefix}_single_{offset:04d}"
        if prediction_path(run_id).exists():
            raise FileExistsError(f"Refusing to overwrite Phase 6 run {run_id}")
        systems["single_agent"].append(
            await run_single_agent_batch(ids, offset, run_id)
        )
    report = {
        "selected_config_index": None,
        "fedotmas_generated": "not_run_no_qualified_config",
        "batch_offsets": [0, 20, 40],
        "batch_size": BATCH_SIZE,
        "worker_model": WORKER_MODEL,
        "systems": systems,
        "phase5_reference": {
            "status": "not_run_in_phase6_smoke",
            "runner": "scripts/run_sampo_phase_5_smoke.py",
            "role": "diagnostic_reference_only",
        },
        "private_ground_truth_used": False,
        "accuracy_used": False,
        "limits": LIMITS,
    }
    atomic_json(batch_dir / "single_agent_smoke_report.json", report)
    return report


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-dir", type=Path, default=None)
    parser.add_argument("--skip-qualification", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--new-run-prefix", action="store_true")
    parser.add_argument("--prepare-retry-audit", action="store_true")
    parser.add_argument("--single-only-smoke", action="store_true")
    args = parser.parse_args()
    batch_dir = args.batch_dir.resolve() if args.batch_dir else _latest_batch()
    if args.new_run_prefix:
        refresh_run_prefix(batch_dir)
    if args.prepare_retry_audit:
        audit = json.loads(
            (batch_dir / "phase6_leakage_audit.json").read_text(encoding="utf-8")
        )
        print(json.dumps({
            "audit_path": str((batch_dir / "phase6_leakage_audit.json").relative_to(ROOT)),
            "task": audit["manual_inputs"]["task"],
            "meta_system_prompt": audit["manual_inputs"]["meta_system_prompt"],
            "mcp_servers": audit["manual_inputs"]["mcp_servers"],
            "tools": audit["manual_inputs"]["tools"],
            "harness_added_instructions": audit["manual_inputs"]["harness_added_instructions"],
            "execution_limits": audit["execution_limits"],
            "generated_config_count": len(audit["generated_mas_configs"]),
            "manual_input_neutrality_asserted": audit["manual_input_neutrality_asserted"],
            "private_ground_truth_used": False,
        }, ensure_ascii=False, indent=2))
        return
    if args.single_only_smoke:
        report = await single_only_smoke(batch_dir)
        print(json.dumps({
            "batch_dir": str(batch_dir.relative_to(ROOT)),
            "single_agent_smoke_report": str((batch_dir / "single_agent_smoke_report.json").relative_to(ROOT)),
            "single_agent": [
                {key: result.get(key) for key in ("passed", "failures", "model_calls", "tool_calls", "prompt_tokens", "completion_tokens", "runtime_seconds")}
                for result in report["systems"]["single_agent"]
            ],
            "generated_mas": report["fedotmas_generated"],
            "private_ground_truth_used": False,
        }, indent=2))
        return
    qualification = None
    if args.skip_qualification:
        qualification = json.loads(
            (batch_dir / "qualification_report.json").read_text(encoding="utf-8")
        )
    else:
        qualification = await qualify(batch_dir)
    selected = qualification["selected_config_index"]
    smoke_report = None if args.skip_smoke else await smoke(batch_dir, selected)
    print(json.dumps({
        "batch_dir": str(batch_dir.relative_to(ROOT)),
        "qualification_report": str((batch_dir / "qualification_report.json").relative_to(ROOT)),
        "selected_config_index": selected,
        "smoke_report": str((batch_dir / "smoke_report.json").relative_to(ROOT)) if smoke_report else None,
        "smoke_reliability": None if smoke_report is None else {
            name: [
                {key: result.get(key) for key in ("passed", "failures", "model_calls", "tool_calls", "prompt_tokens", "completion_tokens", "runtime_seconds")}
                for result in runs
            ]
            for name, runs in smoke_report["systems"].items()
            if name != "deterministic_tfidf"
        },
        "private_ground_truth_used": False,
    }, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
