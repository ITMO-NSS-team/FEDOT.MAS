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

def test_artifacts_deterministic_compact_and_complete(data, monkeypatch):
    monkeypatch.setattr(server, "METHOD_ALIASES", {"alias_one": "one"})
    one=server.prepare_candidate_batch(0,100,["one","two"],5,"rrf"); two=server.prepare_candidate_batch(0,100,["two","one"],5,"rrf"); duplicate=server.prepare_candidate_batch(0,100,["one","two","one"],5,"rrf"); alias=server.prepare_candidate_batch(0,100,["alias-one","two"],5,"rrf"); three=server.prepare_candidate_batch(0,100,["one","two"],4,"rrf")
    assert one["artifact_id"] == two["artifact_id"] == duplicate["artifact_id"] == alias["artifact_id"] != three["artifact_id"]
    assert len(json.dumps(one).encode()) < 64 * 1024 and "method_candidates" not in json.dumps(one)
    evidence=server.get_candidate_evidence(one["artifact_id"],["1"])["examples"][0]
    assert evidence["method_candidates"] and len(evidence["fused_candidates"]) == 5
    assert {x["label"] for x in evidence["fused_candidates"]} <= set(server._labels())

def test_public_method_aliases_are_canonical():
    assert server.METHOD_ALIASES["lexical"] == "bm25_token"
    assert server.METHOD_ALIASES["tfidf_char_ngrams"] == "char_tfidf"

def test_evidence_response_budget_preserves_artifact(data):
    data[0]["raw_work_name"] = "x" * (server.MAX_EVIDENCE_RESPONSE_BYTES + 1)
    artifact = server.prepare_candidate_batch(0, 1, ["one"], 5, "rrf")["artifact_id"]
    with pytest.raises(ValueError, match="48 KiB"):
        server.get_candidate_evidence(artifact, ["1"])
    assert (server.ARTIFACTS / f"{artifact}.json").exists()

def test_diverse_evidence_is_bounded_deterministic_and_uses_artifact_indices(data, monkeypatch):
    def one(examples, labels, k):
        return [[("alpha", .9), ("beta", .8), ("gamma", .7), ("delta", .6), ("epsilon", .5)][:k] for _ in examples]
    def two(examples, labels, k):
        return [[("delta", .9), ("gamma", .8), ("beta", .7), ("alpha", .6), ("epsilon", .5)][:k] for _ in examples]
    monkeypatch.setattr(server, "RETRIEVERS", {"one": one, "two": two})
    artifact = server.prepare_candidate_batch(0, 1, ["one", "two"], 5, "rrf")["artifact_id"]
    full = server.get_candidate_evidence(artifact, ["1"])["examples"][0]
    diverse = server.get_candidate_evidence(artifact, ["1"], candidate_limit=4, selection="diverse_round_robin")
    again = server.get_candidate_evidence(artifact, ["1"], candidate_limit=4, selection="diverse_round_robin")
    candidates = diverse["examples"][0]["fused_candidates"]
    assert diverse == again
    assert diverse["candidate_selection"] == "diverse_round_robin"
    assert [item["label"] for item in candidates] == ["alpha", "delta", "beta", "gamma"]
    assert diverse["examples"][0]["method_candidates"] == {}
    assert set(diverse["examples"][0]["fused_candidates"][0]) == {"candidate_index", "label", "support_count", "best_rank", "methods"}
    assert {item["candidate_index"] for item in candidates} <= {item["candidate_index"] for item in full["fused_candidates"]}
    assert len(candidates) == 4
    indices = [item["candidate_index"] for item in (candidates[1], candidates[0], candidates[2])]
    server.save_review_decisions("diverse", artifact, [{"example_id": "1", "candidate_indices": indices}])
    saved = json.loads(server._run_path("diverse").read_text())
    assert [saved[f"top_{rank}"] for rank in range(1, 4)] == ["delta", "alpha", "beta"]
    with pytest.raises(ValueError, match="requires candidate_limit"):
        server.get_candidate_evidence(artifact, ["1"], selection="diverse_round_robin")
    with pytest.raises(ValueError, match="candidate_limit"):
        server.get_candidate_evidence(artifact, ["1"], candidate_limit=31)

def test_partition_candidate_batch_is_deterministic(data, monkeypatch):
    def one(examples, labels, k):
        return [[("alpha", .9), ("beta", .8), ("gamma", .7)][:k] for _ in examples]
    def two(examples, labels, k):
        return [[("beta", .9), ("alpha", .8), ("gamma", .7)][:k] for _ in examples]
    monkeypatch.setattr(server, "RETRIEVERS", {"one": one, "two": two})
    artifact = server.prepare_candidate_batch(0, 2, ["one", "two"], 3, "rrf")["artifact_id"]
    partition = server.partition_candidate_batch(artifact, ["1", "2"])
    assert partition["review_ids"] == ["1", "2"]
    assert partition["fallback_ids"] == []
    assert partition["review_count"] + partition["fallback_count"] == 2

def test_save_by_ids_review_indices_status_and_finalization(data):
    artifact=server.prepare_candidate_batch(0,100,["one","two"],5,"borda")["artifact_id"]
    assert server.save_candidate_predictions("run",artifact,["1"])["saved"] == 1
    assert server.save_review_decisions("run",artifact,[{"example_id":"1","candidate_indices":[2,1,0]}])["saved"] == 1
    with pytest.raises(ValueError): server.save_review_decisions("run",artifact,[{"example_id":"2","candidate_indices":[0,0,1]}])
    with pytest.raises(ValueError): server.save_review_decisions("run",artifact,[{"example_id":"2","candidate_indices":[0,1,99]}])
    status=server.get_run_status("run",3); assert status["missing_count"] == 99 and len(status["next_missing_ids"]) == 3
    server.save_candidate_predictions("run",artifact,[str(i) for i in range(2,101)])
    assert server.finalize_predictions("run")["examples"] == 100

def test_review_decision_schema_requires_concrete_indices(data):
    schema = server.ReviewDecision.model_json_schema()
    assert set(schema["required"]) == {"example_id", "candidate_indices"}
    indices = schema["properties"]["candidate_indices"]
    assert indices["minItems"] == indices["maxItems"] == 3
    with pytest.raises(Exception):
        server.ReviewDecision(example_id="1", candidate_indices=[0, 1])

def test_prediction_status_is_bounded_batch_only(data):
    artifact=server.prepare_candidate_batch(0,3,["one"],5,"rrf")["artifact_id"]
    assert server.get_prediction_status("run",["1","2"])=={"requested_count":2,"stored_ids":[],"missing_ids":["1","2"],"complete":False}
    server.save_candidate_predictions("run",artifact,["1","2"])
    assert server.get_prediction_status("run",["1","2"])=={"requested_count":2,"stored_ids":["1","2"],"missing_ids":[],"complete":True}
    with pytest.raises(ValueError): server.get_prediction_status("run",["101"])
    with pytest.raises(ValueError): server.get_prediction_status("run",[str(i) for i in range(101)])

def test_staging_is_not_final_prediction_state(data):
    artifact=server.prepare_candidate_batch(0,2,["one"],5,"rrf")["artifact_id"]
    assert server.stage_candidate_predictions("run",artifact,["1","2"])["staged"] == 2
    assert not server.get_prediction_status("run",["1","2"])["complete"]
    with pytest.raises(ValueError): server.stage_candidate_predictions("run",artifact,["1"])

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
