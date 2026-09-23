"""Harness-owned Phase 5 full-pilot evaluation for a selected saved config."""
from __future__ import annotations

import csv
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from sampo_phase_5_policy import evaluate_policy_conformance
from sampo_phase_5_trace import read_trace
BATCH = Path(os.environ["PHASE5_BATCH"]).resolve() if os.environ.get("PHASE5_BATCH") else (
    ROOT / "artifacts/sampo_phase_5/structural_review/batch_1e6819ad6b0a40dda61ab7c86cc18edf"
)
CONFIG = BATCH / f"config_{int(os.environ.get('PHASE5_CONFIG', '1')):02d}"
RUN_ID = os.environ.get("PHASE5_RUN_ID", f"phase5_full_{uuid.uuid4().hex}")
OUT = ROOT / "artifacts/sampo_phase_5" / f"full_{RUN_ID}"
SERVER_PYTHON = ROOT / "mcp-servers/sampo-benchmark/.venv/bin/python"
REQUIRED_TASK_POLICY = ("Request evidence for each review ID exactly once", "never retry get_candidate_evidence")


def pilot_rows() -> list[dict[str, str]]:
    with (ROOT / "artifacts/sampo_benchmark/pilot_inputs.csv").open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def prediction_path() -> Path:
    return ROOT / f"artifacts/sampo_benchmark/mas_runs/{RUN_ID}.jsonl"


def stored_ids() -> set[str]:
    path = prediction_path()
    if not path.exists():
        return set()
    for _ in range(20):
        try:
            return {json.loads(line)["example_id"] for line in path.open() if line.strip()}
        except (UnicodeDecodeError, json.JSONDecodeError):
            time.sleep(0.05)
    raise RuntimeError(f"Prediction file remained unreadable during status check: {path}")


def duplicate_persistence(raw: dict, allowed_ids: set[str]) -> bool:
    candidate_ids: set[str] = set()
    review_decisions: dict[str, tuple[int, ...]] = {}
    for call in raw.get("tool_calls", []):
        if call.get("tool") == "save_candidate_predictions":
            ids = set(call.get("ids") or [])
            if not ids <= allowed_ids:
                continue
            if ids & candidate_ids:
                return True
            candidate_ids.update(ids)
        elif call.get("tool") == "save_review_decisions":
            for decision in call.get("decisions") or []:
                example_id = decision.get("example_id")
                indices = tuple(decision.get("candidate_indices") or [])
                if example_id in review_decisions and review_decisions[example_id] == indices:
                    return True
                review_decisions[example_id] = indices
    return False


def malformed_tool_call(raw: dict, allowed_ids: set[str]) -> bool:
    accepted = {
        "bm25_token", "char_tfidf", "char_word_fusion", "construction_token_tfidf", "word_tfidf",
        "bm25", "bm25_token_ranked", "lexical", "lexical_bm25", "tfidf_char", "tfidf_char_ngrams",
        "char_ngram_tfidf", "tfidf_char_word_hybrid", "char_word_hybrid", "tfidf_construction_token",
        "construction_tfidf", "tfidf_word", "tfidf_word_ngrams",
    }
    for call in raw.get("tool_calls", []):
        if call.get("tool") == "prepare_candidate_batch":
            methods = (call.get("retrieval") or {}).get("methods") or []
            if any(method not in accepted for method in methods):
                return True
        if call.get("tool") == "save_candidate_predictions" and not set(call.get("ids") or []) <= allowed_ids:
            return True
        if call.get("tool") == "save_review_decisions":
            if os.environ.get("PHASE5_EXPERIMENTAL_TAIL") == "1" and not call.get("fill_retrieval_tail"):
                return True
            for decision in call.get("decisions") or []:
                indices = decision.get("candidate_indices")
                if not isinstance(indices, list) or len(indices) != 3 or len(set(indices)) != 3:
                    return True
        if call.get("tool") == "get_candidate_evidence":
            evidence = call.get("evidence") or {}
            if len(call.get("ids") or []) > 4 or evidence.get("selection") != "fused" or evidence.get("candidate_limit") != 10:
                return True
    return False


def prepare_after_write(raw: dict) -> bool:
    durable_write = False
    for call in raw.get("tool_calls", []):
        if call.get("tool") in {"save_candidate_predictions", "save_review_decisions"}:
            durable_write = True
        if durable_write and call.get("tool") == "prepare_candidate_batch":
            return True
    return False


def write_report(report: dict) -> None:
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


def finalize() -> dict:
    code = (
        "import json; from mcp_sampo_benchmark.server import finalize_predictions; "
        f"print(json.dumps(finalize_predictions({RUN_ID!r})))"
    )
    result = subprocess.run(
        [str(SERVER_PYTHON), "-c", code], cwd=ROOT / "mcp-servers/sampo-benchmark",
        check=True, capture_output=True, text=True,
    )
    return json.loads(result.stdout)


def main() -> None:
    saved_task = (CONFIG / "task.txt").read_text(encoding="utf-8")
    if any(fragment.casefold() not in saved_task.casefold() for fragment in REQUIRED_TASK_POLICY):
        raise RuntimeError("Saved generated task lacks the one-shot evidence policy; regenerate configs before an official run")
    resume = os.environ.get("PHASE5_RESUME") == "1"
    if OUT.exists() and not resume:
        raise RuntimeError(f"Refusing to overwrite {OUT}")
    if prediction_path().exists() and not resume or prediction_path().with_suffix(".staged.jsonl").exists() and not resume:
        raise RuntimeError(f"Run ID {RUN_ID} is not clean")
    OUT.mkdir(parents=True, exist_ok=True)
    rows = pilot_rows()
    if len(rows) != 1000:
        raise RuntimeError("Expected fixed 1,000-example pilot")
    if resume:
        report = json.loads((OUT / "report.json").read_text())
        report["status"] = "resumed"
        report["resumed"] = True
    else:
        report = {
            "config": str(CONFIG.relative_to(ROOT)), "run_id": RUN_ID,
            "private_ground_truth_used": False, "fresh_sessions": True,
            "batch_size": 20, "batches": [], "status": "running",
        }
    for offset in range(0, len(rows), 20):
        assigned = [row["example_id"] for row in rows[offset:offset + 20]]
        if resume:
            batch_stored = set(assigned) & stored_ids()
            if batch_stored == set(assigned):
                continue
            if batch_stored:
                raise RuntimeError(
                    f"Cannot resume contaminated batch at offset {offset}: "
                    f"{len(batch_stored)}/{len(assigned)} IDs are already persisted; "
                    "repair or use a new run ID instead of rerunning this batch"
                )
        before = stored_ids()
        trace_path = OUT / f"batch_{offset:04d}_trace.json"
        sentinel = OUT / f"batch_{offset:04d}_complete.json"
        result_path = OUT / f"batch_{offset:04d}_result.json"
        env = dict(
            os.environ, PHASE5_BATCH=str(BATCH), PHASE5_CONFIG=CONFIG.name.removeprefix("config_"), PHASE5_RUN_ID=RUN_ID,
            PHASE5_ASSIGNED_IDS=json.dumps(assigned), PHASE5_OFFSET=str(offset),
            PHASE5_ENFORCE_COMPLETE="1", PHASE5_COMPLETION_SENTINEL=str(sentinel),
            PHASE5_TRACE_PATH=str(trace_path), PHASE5_RESULT_PATH=str(result_path),
            PHASE5_POLICY_SUFFIX=os.environ.get("PHASE5_POLICY_SUFFIX", " Use only fusion=rrf; the only supported fusion strategy for this run is exactly the literal string rrf. Never invent or substitute another fusion value."),
        )
        started = time.monotonic()
        process = subprocess.Popen(
            [sys.executable, str(ROOT / "scripts/run_sampo_phase_5_qualification.py")],
            cwd=ROOT, env=env, start_new_session=True,
        )
        while process.poll() is None and time.monotonic() - started < 300:
            if set(assigned) <= stored_ids():
                time.sleep(0.2)
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    pass
                break
            time.sleep(0.1)
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        process.wait()
        after = stored_ids()
        raw = read_trace(trace_path)
        policy = evaluate_policy_conformance(raw, assigned)
        item = {
            "offset": offset, "assigned_ids": assigned, "runtime_seconds": time.monotonic() - started,
            "stored_ids": sorted(after & set(assigned)), "outside_ids": sorted((after - before) - set(assigned)),
            "max_prompt_tokens": max((call["prompt_tokens"] for call in raw["model_calls"]), default=0),
            "model_calls": len(raw["model_calls"]), "tool_calls": len(raw["tool_calls"]),
            "duplicate_persistence": duplicate_persistence(raw, set(assigned)), "prepare_after_write": prepare_after_write(raw),
            "malformed_tool_call": malformed_tool_call(raw, set(assigned)), "complete": set(assigned) <= after,
            "policy_conformance": policy,
            "sentinel": sentinel.exists(), "failures": raw["failures"],
        }
        report["batches"].append(item)
        write_report(report)
        if (
            not item["complete"] or item["outside_ids"] or item["duplicate_persistence"]
            or item["prepare_after_write"] or item["malformed_tool_call"] or not policy["pass"]
            or item["max_prompt_tokens"] > 64000 or item["model_calls"] > 30
            or item["tool_calls"] > 60 or item["failures"]
        ):
            report["status"] = "failed"
            report["failure_reason"] = f"batch {offset} failed strict checks"
            write_report(report)
            raise RuntimeError(report["failure_reason"])
    expected = {row["example_id"] for row in rows}
    if stored_ids() != expected:
        raise RuntimeError("Full run does not contain exactly all pilot IDs")
    report["finalization"] = finalize()
    report["status"] = "passed"
    write_report(report)


if __name__ == "__main__":
    main()
