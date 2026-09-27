#!/usr/bin/env python3
"""GT-blind cost-demo runner entry point.

The dry-run command validates the frozen public surface and records runtime
configuration without touching private GT. Live model adapters are deliberately
explicitly configured in the experiment manifest before any paid run.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sampo_cost_demo import OUT, read_csv

NEUTRAL = "Map each assigned historical construction work name to three distinct allowed labels using only the supplied public data and tools. Produce valid top-3 predictions for all assigned IDs."
FEDOT_LABEL = "FEDOT.MAS-executed cost-aware workflow; stable Phase-5-style policy; no policy tuning on this split."
SYSTEMS = ["tfidf", "cheap_single_agent", "fedotmas_cost_aware", "codex"]

def commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=OUT.parents[1]).strip()
    except Exception:
        return None

def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    dry = sub.add_parser("dry-run", help="GT-blind 200-example operational validation")
    dry.add_argument("--system", choices=SYSTEMS, default="tfidf")
    dry.add_argument("--limit", type=int, default=200)
    dry.add_argument("--run-id", default="dryrun")
    final = sub.add_parser("final", help="One-shot full run; refuses an already finalized experiment")
    final.add_argument("--run-id", required=True)
    final.add_argument("--system", choices=SYSTEMS, required=True)
    args = parser.parse_args()
    manifest_path = OUT / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit("Frozen split missing; run scripts/prepare_sampo_cost_demo.py first")
    split = read_csv(OUT / "public_inputs.csv")
    labels = read_csv(OUT / "allowed_target_labels.csv")
    if args.command == "dry-run":
        if not 1 <= args.limit <= 200:
            raise SystemExit("Dry run is limited to 1-200 GT-blind examples")
        rows = split[:args.limit]
        experiment_path = OUT / "experiment_manifest.json"
        if not experiment_path.exists():
            experiment = {
                "experiment_id": "sampo_cost_demo_v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "git_commit": commit(), "frozen_ids": json.loads(manifest_path.read_text())["selected_ids"],
                "batch_size": 20, "fresh_session_per_batch": True,
                "models": {"cheap_single_agent": os.getenv("SAMPO_CHEAP_MODEL"), "fedotmas_cost_aware": os.getenv("SAMPO_CHEAP_MODEL"), "codex": os.getenv("SAMPO_CODEX_MODEL")},
                "providers_and_endpoints": {"provider": os.getenv("SAMPO_PROVIDER"), "endpoint": os.getenv("SAMPO_BASE_URL")},
                "prompts": {"cheap_single_agent": NEUTRAL, "fedotmas_cost_aware": "FEDOT.MAS-executed stable Phase-5-style workflow. Use retrieval, selectively request semantic review where the frozen Phase 5 policy requires it, persist safely, and terminate on completion. " + NEUTRAL, "codex": NEUTRAL},
                "tool_surface": [{"name": "list_allowed_labels", "description": "Return the public allowed target-label list."}, {"name": "retrieve_candidates", "description": "Rank allowed labels against public work names using deterministic lexical retrieval."}],
                "limits": {"batch_size": 20, "max_examples_per_dry_run": 200, "session_per_batch": True},
                "pricing_file": "artifacts/sampo_cost_demo/pricing.json", "private_ground_truth_exposed": False,
                "codex_audit": "Record exact system/user prompts and function schemas in each run's transcript.jsonl.",
            }
            experiment_path.write_text(json.dumps(experiment, ensure_ascii=False, indent=2) + "\n")
        output = OUT / "dry_runs" / args.run_id
        output.mkdir(parents=True, exist_ok=False)
        prompts = {"tfidf": "Deterministic TF-IDF; no prompt or tools.", "cheap_single_agent": NEUTRAL, "fedotmas_cost_aware": FEDOT_LABEL + " " + NEUTRAL, "codex": NEUTRAL}
        config = {
            "mode": "gt_blind_dry_run", "system": args.system, "run_id": args.run_id,
            "ids": [r["example_id"] for r in rows], "batch_size": 20, "fresh_session_per_batch": True,
            "model": os.getenv("SAMPO_CODEX_MODEL") if args.system == "codex" else os.getenv("SAMPO_CHEAP_MODEL"),
            "provider": os.getenv("SAMPO_PROVIDER", "configured by environment"), "prompt": prompts[args.system],
            "public_tool_surface": ["list_allowed_labels", "retrieve_candidates"] if args.system != "tfidf" else [],
            "private_ground_truth_opened": False, "accuracy_computed": False,
            "allowed_inspection": ["completion reliability", "tokens", "cost", "runtime", "schema", "tool/runtime failures"],
        }
        (output / "dry_run_manifest.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
        with (output / "assigned_public_inputs.csv").open("x", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["example_id", "raw_work_name"]); writer.writeheader(); writer.writerows(rows)
        print(output)
        return
    finalized = OUT / "finalized.json"
    if finalized.exists():
        raise SystemExit("Final evaluation is already finalized; use a new experiment ID")
    raise SystemExit("Final execution requires frozen provider adapters; use GT-blind dry-run first. No prediction was started.")

if __name__ == "__main__":
    main()
