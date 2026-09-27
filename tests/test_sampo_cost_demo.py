from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import sampo_cost_demo as demo


def test_frozen_split_excludes_pilot_ids_and_normalized_names():
    manifest = json.loads((demo.OUT / "manifest.json").read_text())
    pilot = demo.read_csv(ROOT / "artifacts/sampo_benchmark/pilot_inputs.csv")
    public = demo.read_csv(demo.OUT / "public_inputs.csv")
    pilot_ids = {r["example_id"] for r in pilot}
    pilot_names = {demo.normalized_name(r["raw_work_name"]) for r in pilot}
    assert len(public) == 1000
    assert not ({r["example_id"] for r in public} & pilot_ids)
    assert not ({demo.normalized_name(r["raw_work_name"]) for r in public} & pilot_names)
    assert manifest["selected_ids"] == [r["example_id"] for r in public]


def test_agents_public_contract_does_not_name_private_file():
    source = (ROOT / "scripts/run_sampo_cost_demo.py").read_text()
    assert "private_ground_truth.csv" not in source
    private = (demo.OUT / "private_ground_truth.csv").resolve()
    for row in demo.read_csv(demo.OUT / "public_inputs.csv"):
        assert str(private) not in row.values()


def test_cost_calculation_with_cached_tokens_and_provider_override():
    pricing = {"models": {"m": {"input_usd_per_1m": 2, "cached_input_usd_per_1m": .5, "output_usd_per_1m": 10}}}
    call = {"model": "m", "input_tokens": 1000, "cached_input_tokens": 400, "output_tokens": 100}
    assert demo.call_cost(call, pricing) == pytest.approx(.0024)
    assert demo.call_cost({**call, "provider_cost_usd": .12}, pricing) == .12


def test_failures_stay_in_end_to_end_denominator():
    accuracy, conditional = demo.end_to_end_accuracy(["1", "2"], {"1": "a", "2": "b"}, {"1": {"top_1": "a", "completed": True}})
    assert accuracy == .5
    assert conditional == 1


def test_system_names_are_common_and_runner_exposes_only_smoke():
    runner = (ROOT / "scripts/run_sampo_cost_demo.py").read_text()
    for name in ("tfidf", "cheap_single_agent", "fedotmas_cost_aware", "codex"):
        assert name in runner
    assert 'sub.add_parser("smoke")' in runner
    assert 'sub.add_parser("final")' not in runner


def test_preregistration_contains_fixed_primary_success_threshold():
    prereg = json.loads((demo.OUT / "preregistration.json").read_text())
    assert prereg["success_criterion"]["fedotmas_top1_minimum_codex_top1_minus_pp"] == 3.0
    assert prereg["success_criterion"]["fedotmas_inference_cost_at_most_fraction_of_codex"] == .5
