from __future__ import annotations
import json, sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mcp_sampo_benchmark import server

@pytest.fixture
def data(monkeypatch, tmp_path):
    rows=[{"example_id":str(i),"raw_work_name":f"work {i}"} for i in range(1, 101)]
    monkeypatch.setattr(server,"_inputs",lambda filename="benchmark_inputs.csv":rows)
    monkeypatch.setattr(server,"_labels",lambda:["alpha","beta","gamma","delta","epsilon"])
    monkeypatch.setattr(server,"ARTIFACTS",tmp_path / "artifacts"); monkeypatch.setattr(server,"PUBLIC",tmp_path)
    def rank(examples, labels, k): return [[("alpha",.9),("beta",.8),("gamma",.7),("delta",.6),("epsilon",.5)][:k] for _ in examples]
    monkeypatch.setattr(server,"RETRIEVERS",{"one":rank,"two":rank})
    return rows

def test_artifacts_deterministic_compact_and_complete(data):
    one=server.prepare_candidate_batch(0,100,["one","two"],5,"rrf"); two=server.prepare_candidate_batch(0,100,["two","one"],5,"rrf"); three=server.prepare_candidate_batch(0,100,["one","two"],4,"rrf")
    assert one["artifact_id"] == two["artifact_id"] != three["artifact_id"]
    assert len(json.dumps(one).encode()) < 64 * 1024 and "method_candidates" not in json.dumps(one)
    evidence=server.get_candidate_evidence(one["artifact_id"],["1"])["examples"][0]
    assert evidence["method_candidates"] and len(evidence["fused_candidates"]) == 5
    assert {x["label"] for x in evidence["fused_candidates"]} <= set(server._labels())

def test_save_by_ids_review_indices_status_and_finalization(data):
    artifact=server.prepare_candidate_batch(0,100,["one","two"],5,"borda")["artifact_id"]
    assert server.save_candidate_predictions("run",artifact,["1"])["saved"] == 1
    assert server.save_review_decisions("run",artifact,[{"example_id":"1","candidate_indices":[2,1,0]}])["saved"] == 1
    with pytest.raises(ValueError): server.save_review_decisions("run",artifact,[{"example_id":"2","candidate_indices":[0,0,1]}])
    with pytest.raises(ValueError): server.save_review_decisions("run",artifact,[{"example_id":"2","candidate_indices":[0,1,99]}])
    status=server.get_run_status("run",3); assert status["missing_count"] == 99 and len(status["next_missing_ids"]) == 3
    server.save_candidate_predictions("run",artifact,[str(i) for i in range(2,101)])
    assert server.finalize_predictions("run")["examples"] == 100

def test_old_bulk_api_not_exposed_or_private():
    for name in ("retrieve_candidates","get_allowed_labels","get_input_batch","get_pilot_input_batch","save_prediction_batch","replace_prediction_batch","import_prediction_file"): assert not hasattr(server,name)
    assert "private" not in Path(server.__file__).read_text().casefold()

def test_no_llm_full_pilot_bounded_artifacts(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "ARTIFACTS", tmp_path / "artifacts")
    methods = sorted(server.RETRIEVERS)
    ids, sizes = [], []
    for offset in range(0, 1000, 100):
        response = server.prepare_candidate_batch(offset, 100, methods, 5, "rrf")
        assert (server.ARTIFACTS / f"{response['artifact_id']}.json").exists()
        ids.extend(item["example_id"] for item in response["examples"])
        sizes.append(len(json.dumps(response).encode()))
    expected = [row["example_id"] for row in server._inputs("pilot_inputs.csv")]
    assert len(expected) == 1000 and ids == expected and len(set(ids)) == 1000
    assert max(sizes) < 64 * 1024
