"""Batch-scoped SAMPO tools over explicit public files only."""
from __future__ import annotations
import csv, hashlib, json, os, re, sys, fcntl
from pathlib import Path
from typing import Any
from fastmcp import FastMCP
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts"))
from sampo_baselines import bm25_token_ranked, tfidf_char_ngrams_ranked, tfidf_char_word_hybrid_ranked, tfidf_construction_token_ranked, tfidf_word_ranked

mcp = FastMCP("sampo-cost-demo")
METHODS = {"bm25_token": bm25_token_ranked, "char_tfidf": tfidf_char_ngrams_ranked, "char_word_fusion": tfidf_char_word_hybrid_ranked, "construction_token_tfidf": tfidf_construction_token_ranked, "word_tfidf": tfidf_word_ranked}
FUSIONS = {"rrf", "borda"}

def _scope() -> tuple[list[dict[str, str]], list[str], list[str], str, Path]:
    ids = [x for x in os.environ.get("SAMPO_ASSIGNED_IDS", "").split(",") if x]
    run_id = os.environ.get("SAMPO_RUN_ID", "")
    input_path = Path(os.environ.get("SAMPO_PUBLIC_INPUTS", "")).resolve()
    labels_path = Path(os.environ.get("SAMPO_ALLOWED_LABELS", "")).resolve()
    out = Path(os.environ.get("SAMPO_RUN_DIR", "")).resolve()
    if not ids or len(ids) > 20 or len(set(ids)) != len(ids): raise ValueError("Assigned scope must contain 1-20 unique IDs")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", run_id): raise ValueError("Invalid run ID")
    if "private_ground_truth" in str(input_path).casefold() or "private_ground_truth" in str(labels_path).casefold(): raise ValueError("Private data paths are forbidden")
    if not input_path.is_file() or not labels_path.is_file(): raise ValueError("Public input files are unavailable")
    with input_path.open(encoding="utf-8", newline="") as f: all_rows = list(csv.DictReader(f))
    rows_by_id = {r["example_id"]: r for r in all_rows}
    if not set(ids) <= rows_by_id.keys(): raise ValueError("Assigned IDs are outside configured public input")
    with labels_path.open(encoding="utf-8", newline="") as f: labels = [r["target_label"] for r in csv.DictReader(f)]
    out.mkdir(parents=True, exist_ok=True)
    return [rows_by_id[i] for i in ids], labels, ids, run_id, out

def _artifact_dir() -> Path:
    _, _, _, run_id, out = _scope()
    return out / ".candidate_artifacts" / run_id

def _artifact(artifact_id: str) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{64}", artifact_id): raise ValueError("Invalid artifact ID")
    path = _artifact_dir() / f"{artifact_id}.json"
    if not path.is_file(): raise ValueError("Unknown artifact for this batch")
    return json.loads(path.read_text(encoding="utf-8"))

def _pred_path(out: Path) -> Path: return out / ".predictions.jsonl"

class RankedTop3(BaseModel):
    example_id: str = Field(description="ID from the currently assigned batch")
    candidate_indices: list[int] = Field(min_length=3, max_length=3, description="Three distinct candidate indices in final rank order")

@mcp.tool
def list_methods() -> dict[str, Any]:
    """List supported retrieval methods, fusion choices, public labels and batch limits."""
    _, labels, _, _, _ = _scope()
    return {"methods": sorted(METHODS), "fusion_choices": sorted(FUSIONS), "allowed_target_labels": labels, "max_batch_size": 20, "max_candidates": len(labels)}

@mcp.tool
def prepare_candidates(methods: list[str], k: int, fusion: str) -> dict[str, Any]:
    """Rank the assigned public work names against allowed labels using selected methods and fusion."""
    rows, labels, ids, run_id, out = _scope()
    methods = list(dict.fromkeys(methods))
    if not methods or any(m not in METHODS for m in methods): raise ValueError("Unsupported retrieval method")
    if not 1 <= k <= 50 or fusion not in FUSIONS: raise ValueError("k must be 1-50 and fusion must be advertised")
    params = {"ids": ids, "methods": sorted(methods), "k": k, "fusion": fusion}
    artifact_id = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()
    path = _artifact_dir() / f"{artifact_id}.json"
    if path.exists(): return {"artifact_id": artifact_id, "parameters": params, "examples": json.loads(path.read_text())["examples"]}
    raw = [r["raw_work_name"] for r in rows]
    got = {m: METHODS[m](raw, labels, k) for m in methods}
    examples = []
    for idx, row in enumerate(rows):
        votes: dict[str, float] = {}
        evidence: dict[str, list[dict[str, Any]]] = {}
        for method in methods:
            for rank, (label, score) in enumerate(got[method][idx], 1):
                if label not in labels: raise ValueError("Retriever emitted a non-public label")
                votes[label] = votes.get(label, 0) + (1 / (60 + rank) if fusion == "rrf" else k - rank + 1)
                evidence.setdefault(label, []).append({"method": method, "rank": rank, "score": score})
        ranked = sorted(votes, key=lambda label: (-votes[label], label))
        examples.append({"example_id": row["example_id"], "raw_work_name": row["raw_work_name"], "fused_candidates": [{"candidate_index": i, "label": label, "fusion_score": votes[label]} for i, label in enumerate(ranked)], "method_candidates": evidence})
    data = {"artifact_id": artifact_id, "parameters": params, "examples": examples}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data

@mcp.tool
def inspect_candidates(artifact_id: str, example_ids: list[str], candidate_limit: int = 10) -> dict[str, Any]:
    """Return ranked public candidate labels and retrieval evidence for assigned IDs."""
    data = _artifact(artifact_id); _, _, ids, _, _ = _scope()
    if not 1 <= len(example_ids) <= 10 or not set(example_ids) <= set(ids): raise ValueError("IDs must be from this assigned batch")
    if not 1 <= candidate_limit <= 30: raise ValueError("candidate_limit must be 1-30")
    return {"artifact_id": artifact_id, "examples": [{**e, "fused_candidates": e["fused_candidates"][:candidate_limit]} for e in data["examples"] if e["example_id"] in example_ids]}

def _save(rows: list[dict[str, str]]) -> dict[str, Any]:
    public_rows, labels, ids, run_id, out = _scope()
    if not rows or len({r["example_id"] for r in rows}) != len(rows): raise ValueError("Rows must be nonempty with unique IDs")
    known = set(ids); allowed = set(labels)
    for row in rows:
        vals = [row.get(f"top_{i}", "") for i in range(1, 4)]
        if row.get("example_id") not in known or any(v not in allowed for v in vals) or len(set(vals)) != 3: raise ValueError("Invalid scoped top-three prediction")
    path = _pred_path(out)
    with path.open("a+", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0)
            existing = {r["example_id"]: r for r in (json.loads(line) for line in f if line.strip())}
            for row in rows:
                prior = existing.get(row["example_id"])
                if prior is not None and prior != row: raise ValueError("Write-once conflict for existing prediction")
            fresh = [r for r in rows if r["example_id"] not in existing]
            f.seek(0, os.SEEK_END)
            for row in fresh: f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush(); os.fsync(f.fileno())
            return {"stored": len(fresh), "already_identical": len(rows)-len(fresh), "total": len(existing)+len(fresh), "run_id": run_id}
        finally: fcntl.flock(f, fcntl.LOCK_UN)

@mcp.tool
def save_default_top3(run_id: str, artifact_id: str, example_ids: list[str]) -> dict[str, Any]:
    """Persist the retrieval default top three for assigned IDs; exact repeats are idempotent."""
    data = _artifact(artifact_id); _, _, ids, expected_run, _ = _scope()
    if run_id != expected_run or not example_ids or not set(example_ids) <= set(ids): raise ValueError("Run or IDs outside assigned scope")
    by_id = {e["example_id"]: e for e in data["examples"]}
    return _save([{"example_id": i, **{f"top_{n}": by_id[i]["fused_candidates"][n-1]["label"] for n in range(1, 4)}} for i in example_ids])

@mcp.tool
def save_ranked_top3(run_id: str, artifact_id: str, rankings: list[RankedTop3]) -> dict[str, Any]:
    """Persist caller-ranked candidate indices for assigned IDs; exact repeats are idempotent."""
    data = _artifact(artifact_id); _, _, ids, expected_run, _ = _scope()
    if run_id != expected_run: raise ValueError("Run ID outside assigned scope")
    by_id = {e["example_id"]: e for e in data["examples"]}; rows=[]
    for ranking in rankings:
        item = ranking.model_dump() if isinstance(ranking, RankedTop3) else ranking
        eid, inds = item["example_id"], item["candidate_indices"]
        if eid not in ids or len(set(inds)) != 3: raise ValueError("Invalid scoped ranking")
        candidate = by_id[eid]["fused_candidates"]
        lookup = {x["candidate_index"]: x["label"] for x in candidate}
        if any(i not in lookup for i in inds): raise ValueError("Candidate index outside artifact")
        rows.append({"example_id": eid, **{f"top_{n}": lookup[idx] for n, idx in enumerate(inds, 1)}})
    return _save(rows)

@mcp.tool
def get_prediction_status() -> dict[str, Any]:
    """Return assigned IDs, persisted IDs and IDs not yet persisted for this batch."""
    _, _, ids, _, out = _scope(); path = _pred_path(out)
    stored = [json.loads(line)["example_id"] for line in path.read_text().splitlines() if line] if path.exists() else []
    return {"assigned_ids": ids, "stored_ids": stored, "missing_ids": [i for i in ids if i not in stored]}

def main() -> None:
    mcp.run()

if __name__ == "__main__": main()
