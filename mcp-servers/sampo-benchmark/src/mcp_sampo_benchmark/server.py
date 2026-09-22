"""Artifact-backed, public-only SAMPO benchmark MCP server."""
from __future__ import annotations
import csv, fcntl, hashlib, json, re, sys
from collections import defaultdict
from pathlib import Path
from typing import Any
from pydantic import BaseModel, Field
from fastmcp import FastMCP

ROOT = Path(__file__).resolve().parents[4]
PUBLIC = ROOT / "artifacts" / "sampo_benchmark"
ARTIFACTS = PUBLIC / "candidate_artifacts"
sys.path.insert(0, str(ROOT / "scripts"))
from sampo_baselines import bm25_token_ranked, tfidf_char_ngrams_ranked, tfidf_char_word_hybrid_ranked, tfidf_construction_token_ranked, tfidf_word_ranked

mcp = FastMCP("sampo-benchmark")
RETRIEVERS = {"bm25_token": bm25_token_ranked, "char_tfidf": tfidf_char_ngrams_ranked, "char_word_fusion": tfidf_char_word_hybrid_ranked, "construction_token_tfidf": tfidf_construction_token_ranked, "word_tfidf": tfidf_word_ranked}
FUSIONS = {"rrf", "borda"}
MAX_EVIDENCE_RESPONSE_BYTES = 48 * 1024
MAX_EVIDENCE_CANDIDATES = 20
EVIDENCE_SELECTIONS = {"fused", "diverse_round_robin"}
METHOD_ALIASES = {
    "bm25": "bm25_token", "bm25_token_ranked": "bm25_token", "lexical": "bm25_token", "lexical_bm25": "bm25_token",
    "tfidf_char": "char_tfidf", "tfidf_char_ngrams": "char_tfidf", "char_ngram_tfidf": "char_tfidf",
    "tfidf_char_word_hybrid": "char_word_fusion", "char_word_hybrid": "char_word_fusion",
    "tfidf_construction_token": "construction_token_tfidf", "construction_tfidf": "construction_token_tfidf",
    "tfidf_word": "word_tfidf", "tfidf_word_ngrams": "word_tfidf",
}


class ReviewDecision(BaseModel):
    """One final, artifact-local review decision."""

    example_id: str = Field(description="Artifact example ID to persist")
    candidate_indices: list[int] = Field(
        min_length=3,
        max_length=3,
        description="Exactly three distinct artifact-local candidate_index values, in final rank order",
    )

def _inputs(filename="benchmark_inputs.csv"):
    if filename not in {"benchmark_inputs.csv", "pilot_inputs.csv"}: raise ValueError("Only public benchmark or pilot inputs are available")
    with (PUBLIC / filename).open(encoding="utf-8", newline="") as f: return list(csv.DictReader(f))
def _labels():
    with (PUBLIC / "allowed_target_labels.csv").open(encoding="utf-8", newline="") as f: return [r["target_label"] for r in csv.DictReader(f)]
def _norm(value): return re.sub(r"[^\w]+", "", value.casefold(), flags=re.UNICODE)
def _artifact_path(artifact_id):
    if not re.fullmatch(r"[0-9a-f]{64}", artifact_id): raise ValueError("Invalid artifact_id")
    return ARTIFACTS / f"{artifact_id}.json"
def _artifact(artifact_id):
    path = _artifact_path(artifact_id)
    if not path.exists(): raise ValueError("Unknown candidate artifact")
    return json.loads(path.read_text(encoding="utf-8"))
def _run_path(run_id):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", run_id): raise ValueError("Invalid run_id")
    return PUBLIC / "mas_runs" / f"{run_id}.jsonl"
def _stage_path(run_id): return _run_path(run_id).with_suffix('.staged.jsonl')
def _valid(rows):
    expected, labels = {r["example_id"] for r in _inputs("pilot_inputs.csv")}, set(_labels())
    if len({r.get("example_id") for r in rows}) != len(rows): raise ValueError("Batch contains duplicate IDs")
    for row in rows:
        values = [row.get(f"top_{n}") for n in range(1,4)]
        if row.get("example_id") not in expected or len(set(values)) != 3 or any(v not in labels for v in values): raise ValueError("Invalid or non-pilot prediction")
def _store(run_id, rows, replace):
    _valid(rows); path = _run_path(run_id); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX); f.seek(0)
        stored = {r["example_id"]: r for line in f if line.strip() for r in [json.loads(line)]}
        ids = {r["example_id"] for r in rows}
        if not replace and stored.keys() & ids: raise ValueError("Run already contains a prediction ID")
        stored.update({r["example_id"]: r for r in rows}); f.seek(0); f.truncate()
        f.writelines(json.dumps(r, ensure_ascii=False)+"\n" for r in stored.values()); fcntl.flock(f, fcntl.LOCK_UN)
    return {"saved": len(rows), "total": len(stored)}

@mcp.tool
def list_methods() -> dict[str, Any]:
    """List retrieval methods and deterministic rank-only fusion strategies."""
    return {"methods": sorted(RETRIEVERS), "method_aliases": dict(sorted(METHOD_ALIASES.items())), "fusion_strategies": sorted(FUSIONS), "evidence_selections": sorted(EVIDENCE_SELECTIONS), "max_batch_size": 100, "max_k": 50, "max_evidence_candidates": MAX_EVIDENCE_CANDIDATES}

@mcp.tool
def prepare_candidate_batch(offset: int, limit: int, methods: list[str], k: int, fusion: str) -> dict[str, Any]:
    """Keep full retrieval server-side; canonicalize duplicate method names and return compact summaries."""
    if offset < 0 or not 1 <= limit <= 100 or not 1 <= k <= 50: raise ValueError("offset >= 0, limit <= 100, k <= 50")
    methods = [METHOD_ALIASES.get(re.sub(r"[-\s]+", "_", method.casefold()), re.sub(r"[-\s]+", "_", method.casefold())) if isinstance(method, str) else method for method in methods]
    methods = list(dict.fromkeys(methods))
    if not methods or any(m not in RETRIEVERS for m in methods): raise ValueError("methods must be supported methods")
    if fusion not in FUSIONS: raise ValueError("Unknown fusion strategy")
    selected, methods = _inputs("pilot_inputs.csv")[offset:offset+limit], sorted(methods)
    identity = {"pilot": selected, "offset": offset, "limit": limit, "methods": methods, "k": k, "fusion": fusion}
    artifact_id = hashlib.sha256(json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest(); path = _artifact_path(artifact_id)
    if path.exists(): data = _artifact(artifact_id)
    else:
        labels = _labels(); results = {m: RETRIEVERS[m]([r["raw_work_name"] for r in selected], labels, k) for m in methods}; examples=[]
        for i, row in enumerate(selected):
            entries, totals = defaultdict(list), defaultdict(float)
            for method in methods:
                for rank, (label, score) in enumerate(results[method][i], 1):
                    if label not in labels: raise ValueError("Retriever returned non-public label")
                    entries[label].append({"method":method,"rank":rank,"score":score}); totals[label] += 1/(60+rank) if fusion == "rrf" else k-rank+1
            ordered = sorted(entries, key=lambda label:(-totals[label], min(x["rank"] for x in entries[label]), label))
            examples.append({"example_id":row["example_id"],"raw_work_name":row["raw_work_name"],"fused_candidates":[{"candidate_index":n,"label":label,"fusion_score":totals[label]} for n,label in enumerate(ordered)],"method_candidates":{label:entries[label] for label in ordered}})
        data={"artifact_id":artifact_id,"parameters":identity,"examples":examples}; ARTIFACTS.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(data,ensure_ascii=False,separators=(",",":")),encoding="utf-8")
    summaries=[]
    for e in data["examples"]:
        fused=e["fused_candidates"]; method_top1=[next(label for label, es in e["method_candidates"].items() if any(x["method"]==m and x["rank"]==1 for x in es)) for m in methods]
        summaries.append({"example_id":e["example_id"],"fused_top_3":[x["label"] for x in fused[:3]],"method_count":len(methods),"top_1_vote_count":max(method_top1.count(x) for x in set(method_top1)),"distinct_top1_labels":len(set(method_top1)),"top1_top2_margin":fused[0]["fusion_score"]-(fused[1]["fusion_score"] if len(fused)>1 else 0),"exact_title_match":bool(fused and _norm(e["raw_work_name"])==_norm(fused[0]["label"]))})
    return {"artifact_id":artifact_id,"offset":offset,"total_pilot_examples":len(_inputs("pilot_inputs.csv")),"examples":summaries}

def _evidence_candidates(example: dict[str, Any], selection: str, candidate_limit: int | None) -> list[dict[str, Any]]:
    fused = example["fused_candidates"]
    if selection == "fused":
        return fused if candidate_limit is None else fused[:candidate_limit]
    if candidate_limit is None:
        raise ValueError("diverse_round_robin requires candidate_limit")
    by_method_rank: dict[tuple[str, int], dict[str, Any]] = {}
    by_label = {candidate["label"]: candidate for candidate in fused}
    for label, entries in example["method_candidates"].items():
        for entry in entries:
            by_method_rank[(entry["method"], entry["rank"])] = by_label[label]
    selected, seen = [], set()
    methods = sorted({method for method, _ in by_method_rank})
    for rank in range(1, max((rank for _, rank in by_method_rank), default=0) + 1):
        for method in methods:
            candidate = by_method_rank.get((method, rank))
            if candidate and candidate["candidate_index"] not in seen:
                selected.append(candidate)
                seen.add(candidate["candidate_index"])
                if len(selected) == candidate_limit:
                    return selected
    return selected


@mcp.tool
def get_candidate_evidence(artifact_id: str, example_ids: list[str], candidate_limit: int | None = None, selection: str = "fused") -> dict[str, Any]:
    """Return bounded evidence; diverse_round_robin needs an explicit 1--20 candidate limit and preserves artifact-local indices."""
    if not 1 <= len(example_ids) <= 20 or len(set(example_ids)) != len(example_ids): raise ValueError("Provide 1-20 unique example IDs")
    if selection not in EVIDENCE_SELECTIONS: raise ValueError("Unknown evidence selection")
    if candidate_limit is not None and not 1 <= candidate_limit <= MAX_EVIDENCE_CANDIDATES: raise ValueError(f"candidate_limit must be 1--{MAX_EVIDENCE_CANDIDATES}")
    data=_artifact(artifact_id); known={e["example_id"]:e for e in data["examples"]}
    if any(i not in known for i in example_ids): raise ValueError("IDs must belong to artifact")
    examples=[]
    for example_id in example_ids:
        example=known[example_id]
        candidates=_evidence_candidates(example, selection, candidate_limit)
        labels={candidate["label"] for candidate in candidates}
        examples.append({"example_id":example["example_id"],"raw_work_name":example["raw_work_name"],"fused_candidates":candidates,"method_candidates":({label:entries for label,entries in example["method_candidates"].items() if label in labels} if selection == "fused" else {})})
    response={"artifact_id":artifact_id,"candidate_selection":selection,"candidate_limit":candidate_limit,"examples":examples}
    if len(json.dumps(response,ensure_ascii=False,separators=(",", ":")).encode()) > MAX_EVIDENCE_RESPONSE_BYTES:
        raise ValueError("Requested evidence exceeds the 48 KiB response budget; request a smaller subset of example IDs")
    return response

@mcp.tool
def stage_candidate_predictions(run_id: str, artifact_id: str, example_ids: list[str]) -> dict[str,int]:
    """Durably stage fused candidates for review; staged rows are not final predictions."""
    data=_artifact(artifact_id); known={e["example_id"]:e for e in data["examples"]}
    if not example_ids or len(set(example_ids)) != len(example_ids) or any(i not in known for i in example_ids): raise ValueError("IDs must be unique artifact IDs")
    path=_stage_path(run_id); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+',encoding='utf-8') as file:
        fcntl.flock(file,fcntl.LOCK_EX); file.seek(0)
        staged={json.loads(line)['example_id'] for line in file if line.strip()}
        if staged & set(example_ids): raise ValueError('Run already contains a staged candidate ID')
        file.seek(0,2); file.writelines(json.dumps({'example_id':i,'artifact_id':artifact_id},ensure_ascii=False)+'\n' for i in example_ids); fcntl.flock(file,fcntl.LOCK_UN)
    return {'staged':len(example_ids),'total_staged':len(staged)+len(example_ids)}

@mcp.tool
def save_candidate_predictions(run_id: str, artifact_id: str, example_ids: list[str]) -> dict[str,int]:
    """Persist final fused top-three for unreviewed IDs only; never include IDs already saved by review."""
    data=_artifact(artifact_id); known={e["example_id"]:e for e in data["examples"]}
    if not example_ids or len(set(example_ids)) != len(example_ids) or any(i not in known or len(known[i]["fused_candidates"])<3 for i in example_ids): raise ValueError("IDs must be unique artifact IDs with three candidates")
    return _store(run_id,[{"example_id":i,**{f"top_{n}":known[i]["fused_candidates"][n-1]["label"] for n in range(1,4)}} for i in example_ids],False)

@mcp.tool
def save_review_decisions(run_id: str, artifact_id: str, decisions: list[ReviewDecision]) -> dict[str,int]:
    """Durably persist reviewed IDs by required artifact-local indices; never later save candidates for those same IDs."""
    data=_artifact(artifact_id); known={e["example_id"]:e for e in data["examples"]}
    normalized=[d.model_dump() if isinstance(d, ReviewDecision) else d for d in decisions]
    if not normalized or len({d.get("example_id") for d in normalized}) != len(normalized): raise ValueError("Decisions must have unique example IDs")
    rows=[]
    for d in normalized:
        eid, inds=d.get("example_id"),d.get("candidate_indices")
        if eid not in known or not isinstance(inds,list) or len(inds)!=3 or len(set(inds))!=3 or any(not isinstance(x,int) for x in inds): raise ValueError("Each decision needs three distinct candidate indices")
        labels={x["candidate_index"]:x["label"] for x in known[eid]["fused_candidates"]}
        if any(x not in labels for x in inds): raise ValueError("Candidate index does not belong to example")
        rows.append({"example_id":eid,"top_1":labels[inds[0]],"top_2":labels[inds[1]],"top_3":labels[inds[2]]})
    return _store(run_id,rows,True)

@mcp.tool
def get_run_status(run_id: str, limit: int=50) -> dict[str,Any]:
    """Return bounded pilot progress; pass IDs and artifact IDs between agent roles."""
    if not 1 <= limit <= 50: raise ValueError("limit must be 1-50")
    expected=[r["example_id"] for r in _inputs("pilot_inputs.csv")]; path=_run_path(run_id)
    stored={json.loads(line)["example_id"] for line in path.read_text(encoding="utf-8").splitlines() if line} if path.exists() else set(); missing=[i for i in expected if i not in stored]
    return {"pilot_total":len(expected),"stored_count":len(stored),"missing_count":len(missing),"next_missing_ids":missing[:limit],"finalized":path.with_suffix(".csv").exists()}

@mcp.tool
def get_prediction_status(run_id: str, example_ids: list[str]) -> dict[str,Any]:
    """Verify completion of one assigned pilot unit; use this, not whole-run status, per batch."""
    if not 1 <= len(example_ids) <= 100 or len(set(example_ids)) != len(example_ids):
        raise ValueError("Provide 1-100 unique pilot IDs")
    pilot={row["example_id"] for row in _inputs("pilot_inputs.csv")}
    if any(item not in pilot for item in example_ids): raise ValueError("IDs must belong to the pilot")
    path=_run_path(run_id)
    stored={json.loads(line)["example_id"] for line in path.read_text(encoding="utf-8").splitlines() if line} if path.exists() else set()
    found=[item for item in example_ids if item in stored]; missing=[item for item in example_ids if item not in stored]
    return {"requested_count":len(example_ids),"stored_ids":found,"missing_ids":missing,"complete":not missing}

@mcp.tool
def finalize_predictions(run_id: str) -> dict[str,Any]:
    """Finalize only when every fixed-pilot ID appears exactly once."""
    path=_run_path(run_id); rows=[json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]; expected={r["example_id"] for r in _inputs("pilot_inputs.csv")}
    if len(rows)!=len(expected) or {r["example_id"] for r in rows}!=expected: raise ValueError("Run does not contain every pilot ID exactly once")
    out=path.with_suffix(".csv")
    with out.open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=["example_id","top_1","top_2","top_3"]); writer.writeheader(); writer.writerows(rows)
    try: prediction_path = str(out.relative_to(ROOT))
    except ValueError: prediction_path = str(out)
    return {"prediction_path":prediction_path,"examples":len(rows)}

@mcp.tool
def get_pilot_manifest() -> dict[str,Any]:
    """Return public metadata for the fixed pilot."""
    return json.loads((PUBLIC / "pilot_manifest.json").read_text(encoding="utf-8"))
def main() -> None: mcp.run(show_banner=False)
