#!/usr/bin/env python3
"""Execute cost-demo systems on public IDs only."""
from __future__ import annotations
import argparse, asyncio, csv, hashlib, json, os, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sampo_baselines import tfidf_char_ngrams
from sampo_cost_demo import OUT, ROOT, read_csv
from sampo_cost_runtime import BUDGETS, NEUTRAL_TASK, NEUTRAL_SYSTEM, RuntimeTrace, introspect, model_costs, pricing_preflight, required_runtime, scoped_server, write_system_config

PHASE5_CONFIG = ROOT / "artifacts/sampo_phase_5/structural_review/batch_e3ab204385f84c2085b7b9ae3a86abd3/config_01/config.json"
PHASE5_TASK = (PHASE5_CONFIG.parent / "task.txt").read_text(encoding="utf-8")

def digest(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def commit() -> str: return subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
def runtime_rows(run_dir: Path) -> list[dict[str,str]]:
    path=run_dir/".predictions.jsonl"
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x] if path.exists() else []

def write_outputs(run_dir: Path, system: str, rows: list[dict[str,str]], telemetry: dict, runtime: dict, transcript: list[dict] | None = None) -> None:
    with (run_dir/"predictions.csv").open("x",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["example_id","top_1","top_2","top_3"]);w.writeheader();w.writerows(rows)
    write_system_config(run_dir,system,runtime,telemetry,transcript)

def tfidf_batch(rows: list[dict[str,str]], labels: list[str]) -> list[dict[str,str]]:
    ranked=tfidf_char_ngrams([r["raw_work_name"] for r in rows],labels,3); result=[]
    for row,pred in zip(rows,ranked,strict=True):
        vals=[x for x in pred if x];vals.extend(x for x in labels if x not in vals)
        result.append({"example_id":row["example_id"],**{f"top_{n}":vals[n-1] for n in (1,2,3)}})
    return result

def neutral_user(rows: list[dict[str,str]], labels: list[str]) -> str:
    return NEUTRAL_TASK+"\nAssigned public examples (IDs and work names):\n"+json.dumps(rows,ensure_ascii=False)+"\nAllowed target labels:\n"+json.dumps(labels,ensure_ascii=False)+"\nUse the available tools when useful. Persist valid predictions for every assigned ID."

async def run_agent(system: str, model: str, rows: list[dict[str,str]], labels: list[str], run_id: str, run_root: Path, public_file: Path, phase5: bool=False, batch: int=0) -> dict:
    from google.adk.agents import LlmAgent
    from google.adk.apps.app import App
    from google.adk.sessions import InMemorySessionService
    from fedotmas._settings import resolve_model_config
    from fedotmas.common.llm import make_llm
    from fedotmas.common.llm import _ProxyClient
    from fedotmas.mcp import create_toolset
    from fedotmas.core.runner import run_pipeline
    ids=[r["example_id"] for r in rows]; out=run_root/system/f"batch_{batch:04d}";out.mkdir(parents=True,exist_ok=False)
    registry=scoped_server(ids,run_id,out,public_file,phase5=phase5,batch=batch)
    server_name="sampo-cost-demo-phase5" if phase5 else "sampo-cost-demo"
    description,tools=await introspect(registry,server_name);trace=RuntimeTrace(ids,run_id,out,model,{x["name"] for x in tools},batch=batch)
    _ProxyClient.request_observer=trace.record_provider_request
    if phase5:
        from fedotmas import MAS
        from fedotmas.mas.models import MASConfig
        config=MASConfig.model_validate_json(PHASE5_CONFIG.read_text(encoding="utf-8"))
        config.coordinator.model=model; config.coordinator.tools=[]
        for worker in config.workers: worker.model=model;worker.tools=[server_name]
        mas=MAS(meta_model=model,worker_models=[model],mcp_servers=registry,plugins=[trace.plugin],max_retries=0)
        user="Execute exactly one assigned public batch using the saved Phase 5 configuration. Batch offset "+str(batch)+". Run ID: "+run_id+". Assigned IDs and names: "+json.dumps(rows,ensure_ascii=False)+". Use only these IDs. Allowed labels are available from tools. Return immediately when the assigned batch is durably complete."
        prompt_record={"system_instruction":config.coordinator.instruction+"\n"+"\n".join(w.instruction for w in config.workers),"user_message":user}
        started=time.monotonic()
        try: await mas.build_and_run(config,user,timeout=BUDGETS["max_batch_seconds"])
        except Exception as exc: trace.failures.append(f"execution_error:{type(exc).__name__}:{str(exc)[:300]}")
    else:
        toolset=create_toolset(server_name,registry=registry);instruction=NEUTRAL_SYSTEM
        user=neutral_user(rows,labels)
        prompt_record={"system_instruction":instruction,"user_message":user}
        app=App(name="sampo_cost_demo_agent",root_agent=LlmAgent(name="sampo_cost_demo_agent",model=make_llm(resolve_model_config(model)),instruction=instruction,tools=[toolset]),plugins=[trace.plugin])
        started=time.monotonic()
        try: await run_pipeline(app,user,session_service=InMemorySessionService(),timeout=BUDGETS["max_batch_seconds"])
        except Exception as exc: trace.failures.append(f"execution_error:{type(exc).__name__}:{str(exc)[:300]}")
    duration=time.monotonic()-started;rows_out=runtime_rows(out)
    with (out/"predictions.csv").open("x",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=["example_id","top_1","top_2","top_3"]);writer.writeheader();writer.writerows(rows_out)
    _ProxyClient.request_observer=None
    telemetry=trace.dump();telemetry.update({"model":model,"completed_ids":[r["example_id"] for r in rows_out],"failed_ids":[i for i in ids if i not in {r["example_id"] for r in rows_out}],"completed_examples":len(rows_out),"input_tokens":sum(x["input_tokens"] for x in trace.calls),"cached_input_tokens":sum(x["cached_input_tokens"] for x in trace.calls),"output_tokens":sum(x["output_tokens"] for x in trace.calls),"model_calls_count":len(trace.calls),"mcp_calls_count":len(trace.tools),"runtime_seconds":duration,"safety_ceiling_hit":bool(trace.failures and any("limit" in f for f in trace.failures))})
    pricing=json.loads((OUT/"pricing.json").read_text());telemetry["cost_usd"]=model_costs(telemetry,pricing)
    runtime={"system":system,"model":model,"provider":os.getenv("SAMPO_PROVIDER","openai-compatible"),"endpoint":os.getenv("SAMPO_BASE_URL") or os.getenv("OPENAI_BASE_URL"),"batch_size":len(ids),"budgets":BUDGETS,"fresh_session":True,"prompts":prompt_record,"tool_server_description":description,"tools":tools,"private_gt_path_provided":False,"phase5_config_sha256":digest(PHASE5_CONFIG) if phase5 else None,"git_commit":commit()}
    transcript=[{"type":"exposed_prompt_and_tools","system_instruction":prompt_record["system_instruction"],"user_message":prompt_record["user_message"],"server_description":description,"tools":tools},*trace.transcript]
    write_system_config(out,system,runtime,telemetry,transcript)
    return telemetry

async def get_manifest(rows: list[dict[str,str]], labels: list[str], run_id: str, public_file: Path, cheap: str, strong: str) -> dict:
    ids=[r["example_id"] for r in rows]
    probe=OUT/".schema_probe";probe.mkdir(parents=True,exist_ok=True)
    neutral=scoped_server(ids,run_id,probe,public_file);phase=scoped_server(ids,run_id,probe,public_file,phase5=True)
    _,neutral_tools=await introspect(neutral,"sampo-cost-demo");_,phase_tools=await introspect(phase,"sampo-cost-demo-phase5")
    user=neutral_user(rows,labels)
    phase_config=json.loads(PHASE5_CONFIG.read_text(encoding="utf-8"))
    phase_user=_phase5_user(rows,run_id,0)
    return {"git_commit":commit(),"created_at_utc":datetime.now(timezone.utc).isoformat(),"frozen_test_sha256":digest(OUT/"public_inputs.csv"),"operational_set_sha256":digest(OUT/"operational_inputs.csv"),"models":{"tfidf":None,"cheap_single_agent":cheap,"fedotmas_cost_aware":cheap,"codex":strong},"provider":os.getenv("SAMPO_PROVIDER","openai-compatible"),"endpoint":os.getenv("SAMPO_BASE_URL") or os.getenv("OPENAI_BASE_URL"),"batch_size":20,"budgets":BUDGETS,"prompts":{"cheap_single_agent":{"system":NEUTRAL_SYSTEM,"user":user},"codex":{"system":NEUTRAL_SYSTEM,"user":user},"fedotmas_cost_aware":{"coordinator_instruction":phase_config["coordinator"]["instruction"],"worker_instructions":[w["instruction"] for w in phase_config["workers"]],"user":phase_user,"saved_phase5_task":PHASE5_TASK,"config_sha256":digest(PHASE5_CONFIG)},"one_time_config_generation":{"usd":0,"basis":"Reuse of frozen saved Phase 5 config; no meta-generation during this experiment"}},"mcp_tools":{"neutral":neutral_tools,"phase5_adapter_only_for_fedotmas":phase_tools},"pricing_sha256":digest(OUT/"pricing.json"),"phase5_config_sha256":digest(PHASE5_CONFIG),"private_gt_exposed":False}

def _phase5_user(rows: list[dict[str,str]], run_id: str, offset: int) -> str:
    return "Execute exactly one assigned public batch using the saved Phase 5 configuration. Batch offset "+str(offset)+". Run ID: "+run_id+". Assigned IDs and names: "+json.dumps(rows,ensure_ascii=False)+". Use only these IDs. Allowed labels are available from tools. Return immediately when the assigned batch is durably complete."

def _aggregate_system(system: str, model: str, run_id: str, run_root: Path, all_rows: list[dict[str,str]], batches: list[dict[str,Any]], experiment: dict[str,Any], phase5: bool) -> dict:
    system_dir=run_root/system;system_dir.mkdir(exist_ok=True)
    predictions=[];calls=[];provider_requests=[];mcp_calls=[];worker_calls=[];failures=[];transcript=[];per_batch=[];elapsed=0.0
    for index,batch_rows in enumerate(batches):
        bdir=system_dir/f"batch_{index:04d}"
        if not bdir.exists():
            failures.append({"batch":index,"error":"batch_artifacts_missing"});continue
        predictions.extend(read_csv(bdir/"predictions.csv"))
        telemetry=json.loads((bdir/"telemetry.json").read_text())
        calls.extend(telemetry.get("model_calls",[]));provider_requests.extend(telemetry.get("provider_requests",[]));mcp_calls.extend(telemetry.get("mcp_calls",[]));worker_calls.extend(telemetry.get("worker_calls",[]));failures.extend({"batch":index,"error":x} for x in telemetry.get("failures",[]));elapsed+=telemetry.get("runtime_seconds",0)
        transcript.extend((bdir/"transcript.jsonl").read_text().splitlines())
        per_batch.append(json.loads((bdir/"runtime_manifest.json").read_text()))
    order={r["example_id"]:i for i,r in enumerate(all_rows)}
    predictions.sort(key=lambda r:order[r["example_id"]])
    pred_path=system_dir/"predictions.csv"
    with pred_path.open("x",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=["example_id","top_1","top_2","top_3"]);writer.writeheader();writer.writerows(predictions)
    completed={r["example_id"] for r in predictions};ids=[r["example_id"] for r in all_rows]
    pricing=json.loads((OUT/"pricing.json").read_text());cost=model_costs({"model_calls":calls},pricing)
    telemetry={"system":system,"model":model,"assigned_ids":ids,"completed_ids":[i for i in ids if i in completed],"failed_ids":[i for i in ids if i not in completed],"completed_examples":len(completed),"model_calls":calls,"provider_requests":provider_requests,"provider_request_count":len(provider_requests),"malformed_tool_retry_requests":sum(x["request_kind"]=="malformed_tool_retry" for x in provider_requests),"mcp_calls":mcp_calls,"worker_calls":worker_calls,"tool_calls":len(mcp_calls),"input_tokens":sum(c["input_tokens"] for c in calls),"cached_input_tokens":sum(c["cached_input_tokens"] for c in calls),"output_tokens":sum(c["output_tokens"] for c in calls),"model_calls_count":len(calls),"mcp_calls_count":len(mcp_calls),"cost_usd":cost,"runtime_seconds":elapsed,"safety_ceiling_hit":any("limit" in str(x) for x in failures),"failures":failures}
    runtime={"system":system,"model":model,"batch_size":20,"budgets":BUDGETS,"fresh_session_per_batch":True,"batches":per_batch,"phase5_frozen_config_sha256":digest(PHASE5_CONFIG) if phase5 else None,"git_commit":experiment["git_commit"],"private_ground_truth_path_provided":False}
    (system_dir/"telemetry.json").write_text(json.dumps(telemetry,ensure_ascii=False,indent=2)+"\n")
    (system_dir/"runtime_manifest.json").write_text(json.dumps(runtime,ensure_ascii=False,indent=2)+"\n")
    with (system_dir/"transcript.jsonl").open("x",encoding="utf-8") as f:
        for line in transcript: f.write(line+"\n")
    return telemetry

async def execute(run_id: str, mode: str) -> None:
    cheap,strong,_,_=required_runtime(); pricing_preflight({cheap,strong})
    public_file=OUT/"operational_inputs.csv"
    all_rows=read_csv(public_file); rows=all_rows
    if len(rows)!=20: raise RuntimeError("Operational smoke input must contain exactly 20 examples")
    labels=[x["target_label"] for x in read_csv(OUT/"allowed_target_labels.csv")];run_root=OUT/"runs"/run_id;run_root.mkdir(parents=True,exist_ok=False)
    manifest=await get_manifest(rows,labels,run_id,public_file,cheap,strong)
    if any(manifest["models"][s] is None for s in ("cheap_single_agent","fedotmas_cost_aware","codex")): raise RuntimeError("Runtime model missing")
    with (run_root/"experiment_manifest.json").open("x",encoding="utf-8") as f: f.write(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
    start=time.monotonic();pred=tfidf_batch(rows,labels);elapsed=time.monotonic()-start;d=run_root/"tfidf";d.mkdir()
    telemetry={"system":"tfidf","model":"tfidf_char_ngrams","assigned_ids":[r["example_id"] for r in rows],"completed_ids":[r["example_id"] for r in pred],"failed_ids":[],"completed_examples":len(pred),"model_calls":[],"provider_requests":[],"provider_request_count":0,"malformed_tool_retry_requests":0,"mcp_calls":[],"tool_calls":0,"input_tokens":0,"cached_input_tokens":0,"output_tokens":0,"cost_usd":0.0,"runtime_seconds":elapsed,"method":"tfidf_char_ngrams","safety_ceiling_hit":False,"failures":[]}
    write_outputs(d,"tfidf",pred,telemetry,{"method":"tfidf_char_ngrams","batch_size":20,"git_commit":commit(),"private_gt_path_provided":False})
    for system,model,phase in (("cheap_single_agent",cheap,False),("fedotmas_cost_aware",cheap,True),("codex",strong,False)):
        info=await run_agent(system,model,rows,labels,run_id,run_root,public_file,phase)
        info=_aggregate_system(system,model,run_id,run_root,rows,[rows],manifest,phase)
        esc={i for call in info["mcp_calls"] if call["name"].endswith("get_candidate_evidence") for i in call.get("arguments",{}).get("example_ids",[])} if phase else set()
        print(json.dumps({"system":system,"completed":info["completed_examples"],"model_calls":info["model_calls_count"],"mcp_calls":info["mcp_calls_count"],"input_tokens":info["input_tokens"],"output_tokens":info["output_tokens"],"cost_usd":info["cost_usd"],"runtime_seconds":info["runtime_seconds"],"failures":info["failures"],"fedotmas_escalated_count":len(esc) if phase else None,"fedotmas_escalated_fraction":len(esc)/len(rows) if phase else None},ensure_ascii=False))

def main() -> None:
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest="command",required=True)
    sub.add_parser("preflight");show=sub.add_parser("introspect");show.add_argument("--run-id",default="schema-inspection")
    smoke=sub.add_parser("smoke");smoke.add_argument("--run-id",required=True)
    args=parser.parse_args()
    if args.command=="preflight":
        cheap,strong=os.getenv("SAMPO_CHEAP_MODEL"),os.getenv("SAMPO_CODEX_MODEL")
        pricing=json.loads((OUT/"pricing.json").read_text(encoding="utf-8"));entries=pricing.get("models",{})
        missing=[]
        if not cheap: missing.append("SAMPO_CHEAP_MODEL")
        if not strong: missing.append("SAMPO_CODEX_MODEL")
        if not (os.getenv("SAMPO_BASE_URL") or os.getenv("OPENAI_BASE_URL")): missing.append("provider endpoint")
        for model in (cheap,strong):
            if model and model not in entries: missing.append(f"pricing entry for {model}")
        if not missing:
            try: pricing_preflight({cheap,strong})
            except Exception as exc: missing.append(str(exc))
        result={"models":{"cheap":cheap,"codex":strong},"provider":os.getenv("SAMPO_PROVIDER","openai-compatible"),"endpoint":os.getenv("SAMPO_BASE_URL") or os.getenv("OPENAI_BASE_URL"),"pricing_entries":sorted(entries),"blocking_requirements":missing,"live_inference_ready":not missing}
        print(json.dumps(result,indent=2))
        if missing: raise SystemExit(2)
    elif args.command=="introspect":
        cheap=os.getenv("SAMPO_CHEAP_MODEL") or "<SAMPO_CHEAP_MODEL required>";strong=os.getenv("SAMPO_CODEX_MODEL") or "<SAMPO_CODEX_MODEL required>"
        rows=read_csv(OUT/"operational_inputs.csv")[:20];labels=[x["target_label"] for x in read_csv(OUT/"allowed_target_labels.csv")]
        m=asyncio.run(get_manifest(rows,labels,args.run_id,OUT/"operational_inputs.csv",cheap,strong))
        print(json.dumps(m["prompts"],ensure_ascii=False,indent=2));print(json.dumps(m["mcp_tools"],ensure_ascii=False,indent=2))
    else:
        asyncio.run(execute(args.run_id,"smoke"))
if __name__=="__main__":main()
