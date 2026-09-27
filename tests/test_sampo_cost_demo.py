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


def test_system_names_are_common_and_runner_exposes_all_stages():
    runner = (ROOT / "scripts/run_sampo_cost_demo.py").read_text()
    for name in ("tfidf", "fedotmas_cost_aware", "terra_single_agent"):
        assert name in runner
    assert 'for stage in ("smoke","operational","final"):' in runner
    assert 'sub.add_parser(stage)' in runner


def test_preregistration_contains_fixed_primary_success_threshold():
    prereg = json.loads((demo.OUT / "preregistration.json").read_text())
    assert prereg["success_criterion"]["fedotmas_top1_minimum_terra_top1_minus_pp"] == 3.0
    assert prereg["success_criterion"]["fedotmas_inference_cost_at_most_fraction_of_terra"] == .5
    assert prereg["dropped_comparisons"]["cheap_single_agent"]["reason"] == "experiment simplified before final GT evaluation; primary question is end-to-end FEDOT.MAS vs Terra cost/quality"


def test_operational_200_is_frozen_disjoint_from_smoke_and_final():
    operational = demo.read_csv(demo.OUT / "operational_200_inputs.csv")
    smoke = demo.read_csv(demo.OUT / "operational_inputs.csv")
    final = demo.read_csv(demo.OUT / "public_inputs.csv")
    pilot = demo.read_csv(ROOT / "artifacts/sampo_benchmark/pilot_inputs.csv")
    blocked = [*smoke, *final, *pilot]
    ids = {row["example_id"] for row in blocked}
    names = {demo.normalized_name(row["raw_work_name"]) for row in blocked}
    assert len(operational) == 200
    assert not ({row["example_id"] for row in operational} & ids)
    assert not ({demo.normalized_name(row["raw_work_name"]) for row in operational} & names)
    provenance = json.loads((demo.OUT / "manifest.json").read_text())["operational_200_set"]
    assert provenance["target_size"] == 200
    assert provenance["seed"] == 20260927
    assert provenance["sha256"] == demo.sha256(demo.OUT / "operational_200_inputs.csv")


def test_stage_inputs_and_batch_shapes_are_frozen():
    assert len(demo.read_csv(demo.OUT / "operational_inputs.csv")) == 20
    assert len(demo.read_csv(demo.OUT / "operational_200_inputs.csv")) == 200
    final = demo.read_csv(demo.OUT / "public_inputs.csv")
    assert len(final) == 1000
    assert [len(final[i:i+20]) for i in range(0, len(final), 20)] == [20] * 50


def test_auto_run_id_has_stage_count_and_utc_timestamp():
    import re
    sys.path.insert(0, str(ROOT / "scripts"))
    import run_sampo_cost_demo as runner
    assert re.fullmatch(r"sampo-smoke-20-\d{8}T\d{6}Z(?:-\d+)?", runner.make_run_id("smoke"))
    assert re.fullmatch(r"sampo-operational-200-\d{8}T\d{6}Z(?:-\d+)?", runner.make_run_id("operational"))
    assert re.fullmatch(r"sampo-final-1000-\d{8}T\d{6}Z(?:-\d+)?", runner.make_run_id("final"))


def test_resume_completion_check_requires_same_run_and_all_ids(tmp_path):
    import json
    sys.path.insert(0, str(ROOT / "scripts"))
    import run_sampo_cost_demo as runner
    batch = tmp_path / "batch_0000"
    batch.mkdir()
    ids = ["a", "b"]
    (batch / "telemetry.json").write_text(json.dumps({"run_id":"r","system":"terra_single_agent","failed_ids":[],"completed_examples":2,"failures":[],"cost_complete":True}))
    (batch / "runtime_manifest.json").write_text(json.dumps({"run_id":"r"}))
    (batch / "predictions.csv").write_text("example_id,top_1,top_2,top_3\na,x,y,z\nb,x,y,z\n")
    assert runner.batch_complete(batch, ids, "r", "terra_single_agent")
    assert not runner.batch_complete(batch, ids, "other", "terra_single_agent")
    assert not runner.batch_complete(batch, ["a"], "r", "terra_single_agent")


def test_status_reads_artifacts_without_private_gt(monkeypatch, tmp_path, capsys):
    import json
    sys.path.insert(0, str(ROOT / "scripts"))
    import run_sampo_cost_demo as runner
    out = tmp_path / "artifacts"
    run = out / "runs" / "r"
    run.mkdir(parents=True)
    (run / "experiment_manifest.json").write_text(json.dumps({"run_stage":"smoke","assigned_ids":["a"]}))
    for system in ("tfidf", "fedotmas_cost_aware", "terra_single_agent"):
        d = run / system
        d.mkdir()
        (d / "metrics.json").write_text(json.dumps({"completed_examples":1,"cost_complete":True,"authoritative_total_cost_usd":0.1}))
    monkeypatch.setattr(runner, "OUT", out)
    original = Path.read_text
    def guarded(path, *args, **kwargs):
        assert path.name != "private_ground_truth.csv"
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", guarded)
    assert runner.status("r") == 0
    assert "Ready for evaluation: no" in capsys.readouterr().out
