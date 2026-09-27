"""Phase-5-only adapter over the cost-demo batch scope; not exposed to B/D."""
from __future__ import annotations
from typing import Any
from fastmcp import FastMCP
import server as public

mcp = FastMCP("sampo-cost-demo-phase5")

@mcp.tool
def list_methods() -> dict[str, Any]:
    """Return public retrieval methods, fusion choices, labels and batch limits."""
    return public.list_methods()

@mcp.tool
def prepare_candidate_batch(offset: int, limit: int, methods: list[str], k: int, fusion: str) -> dict[str, Any]:
    """Prepare candidate rankings for the currently assigned batch."""
    expected_offset=int(public.os.environ.get("SAMPO_BATCH","0"))
    if offset != expected_offset: raise ValueError("offset must match the currently assigned batch")
    rows, _, ids, _, _ = public._scope()
    if limit != len(ids): raise ValueError("limit must equal the assigned batch size")
    artifact=public.prepare_candidates(methods, k, fusion)
    compact=[]
    for e in artifact["examples"]:
        method_tops={}
        for label,entries in e["method_candidates"].items():
            for item in entries:
                if item["rank"]==1: method_tops[item["method"]]=label
        candidates=e["fused_candidates"]
        compact.append({"example_id":e["example_id"],"fused_top_3":[c["label"] for c in candidates[:3]],"method_count":len(methods),"top_1_vote_count":max((list(method_tops.values()).count(label) for label in set(method_tops.values())),default=0),"distinct_top1_labels":len(set(method_tops.values())),"top1_top2_margin":(candidates[0]["fusion_score"]-(candidates[1]["fusion_score"] if len(candidates)>1 else 0)) if candidates else 0,"exact_title_match":bool(candidates and public.re.sub(r"[^\w]+","",e["raw_work_name"].casefold())==public.re.sub(r"[^\w]+","",candidates[0]["label"].casefold()))})
    return {"artifact_id":artifact["artifact_id"],"offset":expected_offset,"total_assigned_examples":len(ids),"examples":compact}

@mcp.tool
def partition_candidate_batch(artifact_id: str, example_ids: list[str]) -> dict[str, Any]:
    """Partition assigned IDs by distinct retrieval-method top-1 labels."""
    data=public._artifact(artifact_id); _,_,ids,_,_=public._scope()
    if set(example_ids)!=set(ids): raise ValueError("IDs must cover the assigned batch exactly")
    review=[]; fallback=[]
    for e in data["examples"]:
        tops={label for label, entries in e["method_candidates"].items() if any(x["rank"]==1 for x in entries)}
        (review if len(tops)>1 else fallback).append(e["example_id"])
    return {"review_ids":review,"fallback_ids":fallback}

@mcp.tool
def get_candidate_evidence(artifact_id: str, example_ids: list[str], candidate_limit: int=10, selection: str="fused") -> dict[str, Any]:
    """Return bounded candidate labels and public retrieval evidence for assigned IDs."""
    if selection not in {"fused","diverse_round_robin"}: raise ValueError("Unsupported selection")
    data=public._artifact(artifact_id)
    if selection=="fused": return public.inspect_candidates(artifact_id,example_ids,candidate_limit)
    examples=[]
    for e in data["examples"]:
        if e["example_id"] not in example_ids: continue
        by_method={}
        for label,entries in e["method_candidates"].items():
            for entry in entries: by_method.setdefault(entry["method"],[]).append((entry["rank"],label))
        order=[];seen=set()
        for rank in range(1,max((r for values in by_method.values() for r,_ in values),default=0)+1):
            for method in sorted(by_method):
                label=next((name for r,name in by_method[method] if r==rank),None)
                candidate=next((c for c in e["fused_candidates"] if c["label"]==label),None)
                if candidate and candidate["candidate_index"] not in seen:
                    order.append(candidate);seen.add(candidate["candidate_index"])
                    if len(order)>=candidate_limit: break
            if len(order)>=candidate_limit: break
        compact=[]
        original_rank={c["candidate_index"]:n for n,c in enumerate(e["fused_candidates"],1)}
        for candidate in order:
            entries=e["method_candidates"].get(candidate["label"],[])
            compact.append({"candidate_index":candidate["candidate_index"],"label":candidate["label"],"fused_rank":original_rank[candidate["candidate_index"]],"support_count":len({x["method"] for x in entries}),"best_rank":min((x["rank"] for x in entries),default=None),"methods":sorted({x["method"] for x in entries})})
        examples.append({"example_id":e["example_id"],"raw_work_name":e["raw_work_name"],"fused_candidates":compact,"method_candidates":{}})
    return {"artifact_id":artifact_id,"candidate_selection":selection,"candidate_limit":candidate_limit,"examples":examples}

@mcp.tool
def save_candidate_predictions(run_id: str, artifact_id: str, example_ids: list[str]) -> dict[str, Any]:
    """Persist retrieval-default top-three predictions for assigned IDs."""
    _,_,_,expected,_=public._scope()
    if run_id!=expected: raise ValueError("Run ID mismatch")
    data=public._artifact(artifact_id); lookup={e["example_id"]:e for e in data["examples"]}
    rows=[{"example_id":i,**{f"top_{n}":lookup[i]["fused_candidates"][n-1]["label"] for n in range(1,4)}} for i in example_ids]
    return public._save(rows)

@mcp.tool
def save_review_decisions(run_id: str, artifact_id: str, decisions: list[dict[str, Any]]) -> dict[str, Any]:
    """Persist caller-ranked candidate indices for assigned IDs."""
    _,_,_,expected,_=public._scope()
    if run_id!=expected: raise ValueError("Run ID mismatch")
    data=public._artifact(artifact_id); lookup={e["example_id"]:e for e in data["examples"]}; rows=[]
    for d in decisions:
        eid, inds=d["example_id"],d["candidate_indices"]
        if len(inds)!=3 or len(set(inds))!=3: raise ValueError("Exactly three distinct candidate indices required")
        candidates=lookup[eid]["fused_candidates"]; by_index={x["candidate_index"]:x["label"] for x in candidates}
        if any(i not in by_index for i in inds): raise ValueError("Candidate index outside artifact")
        rows.append({"example_id":eid,**{f"top_{n}":by_index[index] for n,index in enumerate(inds,1)}})
    return public._save(rows)

@mcp.tool
def get_prediction_status() -> dict[str, Any]:
    """Return assigned and persisted prediction IDs."""
    return public.get_prediction_status()

def main() -> None: mcp.run()
if __name__=="__main__": main()
