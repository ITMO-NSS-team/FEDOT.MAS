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

PHASE5_DIR = OUT / "frozen_phase5"
PHASE5_CONFIG = PHASE5_DIR / "config.json"
PHASE5_TASK = (PHASE5_DIR / "task.txt").read_text(encoding="utf-8")

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

def neutral_user(rows: list[dict[str,str]], run_id: str) -> str:
    return NEUTRAL_TASK+"\nAssigned public examples (IDs and work names):\n"+json.dumps(rows,ensure_ascii=False)+"\nPersistence run ID: "+run_id+". Each inspect_candidates call accepts at most 10 example IDs; split larger requests to stay within tool limits.\n"+"Tool-persisted predictions are the completion criterion. A textual answer without durable saved predictions is incomplete. Keep reasoning concise, make the required persistence tool calls, and return only after every assigned ID has been persisted or a real execution failure prevents completion."

def assert_proxy_tree(agent, proxy_type) -> None:
    pending=[agent]
    while pending:
        current=pending.pop()
        if hasattr(current,"model"):
            llm=current.model
            if not isinstance(getattr(llm,"llm_client",None),proxy_type):
                raise RuntimeError(f"Unobserved provider client on agent {getattr(current,'name','unknown')}")
        pending.extend(getattr(current,"sub_agents",[]) or [])

async def run_agent(system: str, model: str, endpoint: str, rows: list[dict[str,str]], labels: list[str], run_id: str, run_root: Path, public_file: Path, phase5: bool=False, batch: int=0) -> dict:
    from google.adk.agents import LlmAgent
    from google.adk.apps.app import App
    from google.adk.sessions import InMemorySessionService
    from fedotmas.common.llm import make_llm
    from fedotmas.common.llm import _ProxyClient
    from fedotmas.mcp import create_toolset
    from fedotmas.core.runner import run_pipeline
    from sampo_cost_runtime import model_config_for
    model_config=model_config_for(model,endpoint)
    if model_config.api_base != endpoint: raise RuntimeError("Model endpoint differs from preflight endpoint")
    ids=[r["example_id"] for r in rows]; out=run_root/system/f"batch_{batch:04d}";out.mkdir(parents=True,exist_ok=False)
    os.environ["FEDOTMAS_LOG_DIR"] = str((out / "logs").resolve())
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
        mas=MAS(meta_model=model_config,worker_models=[model_config],mcp_servers=registry,plugins=[trace.plugin],max_retries=0)
        user="Execute exactly one assigned public batch using the saved Phase 5 configuration. Batch offset "+str(batch)+". Run ID: "+run_id+". Assigned IDs and names: "+json.dumps(rows,ensure_ascii=False)+". Use only these IDs. Allowed labels are available from tools. Return immediately when the assigned batch is durably complete."
        prompt_record={"system_instruction":config.coordinator.instruction+"\n"+"\n".join(w.instruction for w in config.workers),"user_message":user}
        started=time.monotonic()
        try:
            app=mas.build_app(config)
            assert_proxy_tree(app.root_agent,_ProxyClient)
            await run_pipeline(app,user,session_service=InMemorySessionService(),timeout=BUDGETS["max_batch_seconds"])
        except Exception as exc: trace.failures.append(f"execution_error:{type(exc).__name__}:{str(exc)[:300]}")
    else:
        toolset=create_toolset(server_name,registry=registry);instruction=NEUTRAL_SYSTEM
        user=neutral_user(rows,run_id)
        prompt_record={"system_instruction":instruction,"user_message":user}
        llm=make_llm(model_config)
        if not isinstance(getattr(llm,"llm_client",None),_ProxyClient): raise RuntimeError("Configured model did not resolve to observable _ProxyClient")
        app=App(name="sampo_cost_demo_agent",root_agent=LlmAgent(name="sampo_cost_demo_agent",model=llm,instruction=instruction,tools=[toolset]),plugins=[trace.plugin])
        started=time.monotonic()
        try: await run_pipeline(app,user,session_service=InMemorySessionService(),timeout=BUDGETS["max_batch_seconds"])
        except Exception as exc: trace.failures.append(f"execution_error:{type(exc).__name__}:{str(exc)[:300]}")
    duration=time.monotonic()-started;rows_out=runtime_rows(out)
    with (out/"predictions.csv").open("x",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=["example_id","top_1","top_2","top_3"]);writer.writeheader();writer.writerows(rows_out)
    _ProxyClient.request_observer=None
    telemetry=trace.dump();telemetry.update({"model":model,"completed_ids":[r["example_id"] for r in rows_out],"failed_ids":[i for i in ids if i not in {r["example_id"] for r in rows_out}],"completed_examples":len(rows_out),"model_calls_count":len(trace.calls),"mcp_calls_count":len(trace.tools),"runtime_seconds":duration,"safety_ceiling_hit":bool(trace.failures and any("limit" in f for f in trace.failures))})
    pricing=json.loads((OUT/"pricing.json").read_text());telemetry["cost_usd"]=model_costs(telemetry,pricing);telemetry["cost_complete"]=telemetry["cost_usd"] is not None
    runtime={"system":system,"model":model,"provider":os.getenv("SAMPO_PROVIDER","openai-compatible"),"endpoint":endpoint,"model_configuration_path":"ModelConfig(api_base=required_runtime endpoint, api_key=SAMPO_API_KEY or OPENAI_API_KEY) -> make_llm -> _ProxyClient","batch_size":len(ids),"budgets":BUDGETS,"fresh_session":True,"prompts":prompt_record,"tool_server_description":description,"tools":tools,"private_gt_path_provided":False,"phase5_config_sha256":digest(PHASE5_CONFIG) if phase5 else None,"git_commit":commit()}
    transcript=[{"type":"exposed_prompt_and_tools","system_instruction":prompt_record["system_instruction"],"user_message":prompt_record["user_message"],"server_description":description,"tools":tools},*trace.transcript]
    write_system_config(out,system,runtime,telemetry,transcript)
    return telemetry

async def get_manifest(rows: list[dict[str,str]], labels: list[str], run_id: str, public_file: Path, fedot: str, terra: str) -> dict:
    ids=[r["example_id"] for r in rows]
    probe=OUT/".schema_probe";probe.mkdir(parents=True,exist_ok=True)
    neutral=scoped_server(ids,run_id,probe,public_file);phase=scoped_server(ids,run_id,probe,public_file,phase5=True)
    _,neutral_tools=await introspect(neutral,"sampo-cost-demo");_,phase_tools=await introspect(phase,"sampo-cost-demo-phase5")
    user=neutral_user(rows,run_id)
    phase_config=json.loads(PHASE5_CONFIG.read_text(encoding="utf-8"))
    phase_user=_phase5_user(rows,run_id,0)
    return {"git_commit":commit(),"created_at_utc":datetime.now(timezone.utc).isoformat(),"frozen_test_sha256":digest(OUT/"public_inputs.csv"),"operational_set_sha256":digest(OUT/"operational_inputs.csv"),"models":{"tfidf":None,"fedotmas_cost_aware":fedot,"terra_single_agent":terra},"provider":os.getenv("SAMPO_PROVIDER","openai-compatible"),"endpoint":os.getenv("SAMPO_BASE_URL") or os.getenv("OPENAI_BASE_URL"),"batch_size":20,"budgets":BUDGETS,"primary_question":"Can a FEDOT.MAS cost-aware workflow using GPT-5.6 Luna achieve accuracy reasonably close to a strong GPT-5.6 Terra single agent while using substantially less inference cost?","prompts":{"terra_single_agent":{"system":NEUTRAL_SYSTEM,"user":user},"fedotmas_cost_aware":{"coordinator_instruction":phase_config["coordinator"]["instruction"],"worker_instructions":[w["instruction"] for w in phase_config["workers"]],"user":phase_user,"saved_phase5_task":PHASE5_TASK,"config_sha256":digest(PHASE5_CONFIG)},"one_time_config_generation":{"usd":0,"basis":"Reuse of frozen saved Phase 5 config; no meta-generation during this experiment"}},"mcp_tools":{"neutral":neutral_tools,"phase5_adapter_only_for_fedotmas":phase_tools},"pricing_sha256":digest(OUT/"pricing.json"),"phase5_config_sha256":digest(PHASE5_CONFIG),"private_gt_exposed":False}

def _phase5_user(rows: list[dict[str,str]], run_id: str, offset: int) -> str:
    return "Execute exactly one assigned public batch using the saved Phase 5 configuration. Batch offset "+str(offset)+". Run ID: "+run_id+". Assigned IDs and names: "+json.dumps(rows,ensure_ascii=False)+". Use only these IDs. Allowed labels are available from tools. Return immediately when the assigned batch is durably complete."

def _find_list(value: object, key: str) -> list[str]:
    if isinstance(value, dict):
        found=value.get(key)
        if isinstance(found,list) and all(isinstance(x,str) for x in found): return found
        for child in value.values():
            result=_find_list(child,key)
            if result: return result
    elif isinstance(value,list):
        for child in value:
            result=_find_list(child,key)
            if result: return result
    return []

def phase5_partition_metrics(info: dict[str,Any], batch_size: int) -> dict[str,Any]:
    review=set(); fallback=set()
    for call in info.get("mcp_calls",[]):
        if not call.get("name","").endswith("partition_candidate_batch"): continue
        response=call.get("response",{})
        review.update(_find_list(response,"review_ids")); fallback.update(_find_list(response,"fallback_ids"))
    evidence=sorted({example_id for call in info.get("mcp_calls",[]) if call.get("name","").endswith("get_candidate_evidence") for example_id in (call.get("arguments",{}).get("example_ids",[]) if isinstance(call.get("arguments"),dict) else [])})
    return {"review_ids":sorted(review),"fallback_ids":sorted(fallback),"review_count":len(review),"fallback_count":len(fallback),"evidence_requested_ids":evidence,"semantic_review_fraction":len(review)/batch_size if batch_size else 0.0}

def _aggregate_system(system: str, model: str, run_id: str, run_root: Path, all_rows: list[dict[str,str]], batches: list[dict[str,Any]], experiment: dict[str,Any], phase5: bool) -> dict:
    system_dir=run_root/system;system_dir.mkdir(exist_ok=True)
    predictions=[];calls=[];provider_requests=[];mcp_calls=[];worker_calls=[];failures=[];transcript=[];per_batch=[];elapsed=0.0;delegation_attempts=0
    for index,batch_rows in enumerate(batches):
        bdir=system_dir/f"batch_{index:04d}"
        if not bdir.exists():
            failures.append({"batch":index,"error":"batch_artifacts_missing"});continue
        predictions.extend(read_csv(bdir/"predictions.csv"))
        telemetry=json.loads((bdir/"telemetry.json").read_text())
        calls.extend(telemetry.get("model_calls",[]));provider_requests.extend(telemetry.get("provider_requests",[]));mcp_calls.extend(telemetry.get("mcp_calls",[]));worker_calls.extend(telemetry.get("worker_calls",[]));delegation_attempts+=telemetry.get("worker_delegations",0);failures.extend({"batch":index,"error":x} for x in telemetry.get("failures",[]));elapsed+=telemetry.get("runtime_seconds",0)
        transcript.extend((bdir/"transcript.jsonl").read_text().splitlines())
        per_batch.append(json.loads((bdir/"runtime_manifest.json").read_text()))
    for call in mcp_calls:
        response=call.get("response")
        if isinstance(response,dict) and response.get("isError") in (True,"True"):
            messages=[item.get("text","") for item in response.get("content",[]) if isinstance(item,dict)]
            failures.append({"tool":call.get("name"),"error":"; ".join(messages)[:500] or "MCP tool returned isError"})
    order={r["example_id"]:i for i,r in enumerate(all_rows)}
    predictions.sort(key=lambda r:order[r["example_id"]])
    pred_path=system_dir/"predictions.csv"
    with pred_path.open("x",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=["example_id","top_1","top_2","top_3"]);writer.writeheader();writer.writerows(predictions)
    completed={r["example_id"] for r in predictions};ids=[r["example_id"] for r in all_rows]
    pricing=json.loads((OUT/"pricing.json").read_text());cost=model_costs({"model_calls":calls,"model_calls_count":len(calls),"provider_requests":provider_requests,"proxy_observable":True},pricing)
    known=[x for x in provider_requests if x["usage_known"]]
    truncations=sum(c.get("finish_reason") in {"length","max_tokens"} for c in provider_requests)
    provider_cost=sum(float(c["provider_cost_usd"]) for c in provider_requests if c.get("provider_cost_usd") is not None)
    fallback_cost=sum(call_cost(c,pricing) for c in provider_requests if c.get("usage_known") and c.get("provider_cost_usd") is None)
    telemetry={"system":system,"model":model,"assigned_ids":ids,"completed_ids":[i for i in ids if i in completed],"failed_ids":[i for i in ids if i not in completed],"completed_examples":len(completed),"model_calls":calls,"provider_requests":provider_requests,"provider_request_count":len(provider_requests),"malformed_tool_retry_requests":sum(x["request_kind"]=="malformed_tool_retry" for x in provider_requests),"proxy_observable":True,"cost_complete":cost is not None,"mcp_calls":mcp_calls,"worker_calls":worker_calls,"worker_delegations":delegation_attempts,"accepted_worker_delegations":sum(c.get("name","").endswith("construction_batch_specialist") for c in worker_calls),"tool_calls":len(mcp_calls),"input_tokens":sum(c["input_tokens"] for c in known),"uncached_input_tokens":sum(max(0,c["input_tokens"]-c["cached_input_tokens"]) for c in known),"cached_input_tokens":sum(c["cached_input_tokens"] for c in known),"output_tokens":sum(c["output_tokens"] for c in known),"provider_reported_cost_usd":provider_cost,"fallback_calculated_cost_usd":fallback_cost,"finish_reasons":[c.get("finish_reason") for c in provider_requests],"output_truncation_count":truncations,"output_truncation_hit":truncations>0,"adk_input_tokens":sum(c["input_tokens"] for c in calls),"adk_cached_input_tokens":sum(c["cached_input_tokens"] for c in calls),"adk_output_tokens":sum(c["output_tokens"] for c in calls),"model_calls_count":len(calls),"coordinator_model_calls":sum(c.get("agent_name")=="construction_label_batch_coordinator" for c in calls),"worker_model_calls":sum(c.get("agent_name")=="construction_batch_specialist" for c in calls),"mcp_calls_count":len(mcp_calls),"cost_usd":cost,"runtime_seconds":elapsed,"safety_ceiling_hit":any("limit" in str(x) for x in failures),"failures":failures}
    if phase5:
        telemetry["phase5_partition_metrics"]=phase5_partition_metrics({"mcp_calls":mcp_calls},len(ids))
    runtime={"system":system,"model":model,"batch_size":20,"budgets":BUDGETS,"fresh_session_per_batch":True,"batches":per_batch,"phase5_frozen_config_sha256":digest(PHASE5_CONFIG) if phase5 else None,"git_commit":experiment["git_commit"],"private_ground_truth_path_provided":False}
    (system_dir/"telemetry.json").write_text(json.dumps(telemetry,ensure_ascii=False,indent=2)+"\n")
    (system_dir/"runtime_manifest.json").write_text(json.dumps(runtime,ensure_ascii=False,indent=2)+"\n")
    with (system_dir/"transcript.jsonl").open("x",encoding="utf-8") as f:
        for line in transcript: f.write(line+"\n")
    return telemetry

async def execute(run_id: str, mode: str) -> None:
    fedot,terra,_,endpoint=required_runtime(); pricing=pricing_preflight({fedot,terra})
    public_file=OUT/"operational_inputs.csv"
    all_rows=read_csv(public_file); rows=all_rows
    if len(rows)!=20: raise RuntimeError("Operational smoke input must contain exactly 20 examples")
    labels=[x["target_label"] for x in read_csv(OUT/"allowed_target_labels.csv")];run_root=OUT/"runs"/run_id;run_root.mkdir(parents=True,exist_ok=False)
    manifest=await get_manifest(rows,labels,run_id,public_file,fedot,terra)
    with (run_root/"experiment_manifest.json").open("x",encoding="utf-8") as f: f.write(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
    start=time.monotonic();pred=tfidf_batch(rows,labels);elapsed=time.monotonic()-start;d=run_root/"tfidf";d.mkdir()
    telemetry={"system":"tfidf","model":"tfidf_char_ngrams","assigned_ids":[r["example_id"] for r in rows],"completed_ids":[r["example_id"] for r in pred],"failed_ids":[],"completed_examples":len(pred),"model_calls":[],"provider_requests":[],"provider_request_count":0,"malformed_tool_retry_requests":0,"mcp_calls":[],"tool_calls":0,"input_tokens":0,"uncached_input_tokens":0,"cached_input_tokens":0,"output_tokens":0,"cost_usd":0.0,"cost_complete":True,"provider_reported_cost_usd":0.0,"fallback_calculated_cost_usd":0.0,"finish_reasons":[],"output_truncation_count":0,"output_truncation_hit":False,"runtime_seconds":elapsed,"method":"tfidf_char_ngrams","safety_ceiling_hit":False,"failures":[]}
    write_outputs(d,"tfidf",pred,telemetry,{"method":"tfidf_char_ngrams","batch_size":20,"git_commit":commit(),"private_gt_path_provided":False})
    info_by_system={"tfidf":telemetry}; any_failed=False
    for system,model,phase in (("fedotmas_cost_aware",fedot,True),("terra_single_agent",terra,False)):
        try:
            await run_agent(system,model,endpoint,rows,labels,run_id,run_root,public_file,phase)
            info=_aggregate_system(system,model,run_id,run_root,rows,[rows],manifest,phase)
        except Exception as exc:
            any_failed=True
            from fedotmas.common.llm import _ProxyClient
            _ProxyClient.request_observer=None
            system_dir=run_root/system;system_dir.mkdir(exist_ok=True)
            pred=system_dir/"predictions.csv"
            if not pred.exists():
                with pred.open("x",encoding="utf-8",newline="") as stream:
                    csv.DictWriter(stream,fieldnames=["example_id","top_1","top_2","top_3"]).writeheader()
            info={"system":system,"model":model,"assigned_ids":[r["example_id"] for r in rows],"completed_ids":[],"failed_ids":[r["example_id"] for r in rows],"completed_examples":0,"model_calls":[],"provider_requests":[],"provider_request_count":0,"malformed_tool_retry_requests":0,"proxy_observable":True,"cost_complete":False,"mcp_calls":[],"worker_calls":[],"worker_delegations":0,"tool_calls":0,"input_tokens":0,"uncached_input_tokens":0,"cached_input_tokens":0,"output_tokens":0,"provider_reported_cost_usd":None,"fallback_calculated_cost_usd":None,"finish_reasons":[],"output_truncation_count":0,"output_truncation_hit":False,"model_calls_count":0,"coordinator_model_calls":0,"worker_model_calls":0,"mcp_calls_count":0,"cost_usd":None,"runtime_seconds":0,"safety_ceiling_hit":False,"failures":[f"system_error:{type(exc).__name__}:{str(exc)[:500]}"]}
            (system_dir/"telemetry.json").write_text(json.dumps(info,ensure_ascii=False,indent=2)+"\n")
            (system_dir/"runtime_manifest.json").write_text(json.dumps({"system":system,"model":model,"batch_size":len(rows),"private_ground_truth_path_provided":False},indent=2)+"\n")
            (system_dir/"transcript.jsonl").write_text("")
        info_by_system[system]=info
        phase_metrics=phase5_partition_metrics(info,len(rows)) if phase else None
        print(json.dumps({"system":system,"completed":info["completed_examples"],"provider_requests":info["provider_request_count"],"model_calls":info["model_calls_count"],"coordinator_model_calls":info.get("coordinator_model_calls"),"worker_model_calls":info.get("worker_model_calls"),"worker_delegations":info.get("worker_delegations"),"mcp_calls":info["mcp_calls_count"],"partition_metrics":phase_metrics,"input_tokens":info["input_tokens"],"uncached_input_tokens":info.get("uncached_input_tokens"),"cached_input_tokens":info["cached_input_tokens"],"output_tokens":info["output_tokens"],"cost_usd":info["cost_usd"],"provider_reported_cost_usd":info.get("provider_reported_cost_usd"),"fallback_calculated_cost_usd":info.get("fallback_calculated_cost_usd"),"cost_complete":info.get("cost_complete"),"runtime_seconds":info["runtime_seconds"],"finish_reasons":info.get("finish_reasons"),"truncations":info.get("output_truncation_count"),"safety_ceiling_hit":info.get("safety_ceiling_hit"),"failures":info["failures"]},ensure_ascii=False))
        if info["completed_examples"]!=20 or not info["cost_complete"] or info["output_truncation_hit"] or info["safety_ceiling_hit"] or info["failures"]:
            any_failed=True
    ratio=info_by_system["terra_single_agent"].get("cost_usd",0)/info_by_system["fedotmas_cost_aware"].get("cost_usd",0) if info_by_system["fedotmas_cost_aware"].get("cost_usd") else None
    ft=info_by_system["fedotmas_cost_aware"]; tt=info_by_system["terra_single_agent"]
    token_ratio=(tt.get("input_tokens",0)+tt.get("output_tokens",0))/(ft.get("input_tokens",0)+ft.get("output_tokens",0)) if (ft.get("input_tokens",0)+ft.get("output_tokens",0)) else None
    print(json.dumps({"gt_blind_operational_ratios":{"terra_cost_over_fedotmas_cost":ratio,"terra_tokens_over_fedotmas_tokens":token_ratio}},ensure_ascii=False))
    if any_failed: raise SystemExit(1)

def main() -> None:
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest="command",required=True)
    sub.add_parser("preflight");show=sub.add_parser("introspect");show.add_argument("--run-id",default="schema-inspection")
    smoke=sub.add_parser("smoke");smoke.add_argument("--run-id",required=True)
    args=parser.parse_args()
    if args.command=="preflight":
        fedot,terra=os.getenv("SAMPO_FEDOT_MODEL"),os.getenv("SAMPO_TERRA_MODEL")
        pricing=json.loads((OUT/"pricing.json").read_text(encoding="utf-8"));entries=pricing.get("models",{})
        missing=[]
        if fedot != "openai/gpt-5.6-luna": missing.append("SAMPO_FEDOT_MODEL=openai/gpt-5.6-luna")
        if terra != "openai/gpt-5.6-terra": missing.append("SAMPO_TERRA_MODEL=openai/gpt-5.6-terra")
        if not (os.getenv("SAMPO_BASE_URL") or os.getenv("OPENAI_BASE_URL")): missing.append("provider endpoint")
        if not (os.getenv("SAMPO_API_KEY") or os.getenv("OPENAI_API_KEY")): missing.append("SAMPO_API_KEY/OPENAI_API_KEY")
        for model in (fedot,terra):
            if model and model not in entries: missing.append(f"pricing entry for {model}")
        if not missing:
            try: pricing_preflight({fedot,terra})
            except Exception as exc: missing.append(str(exc))
        result={"models":{"fedotmas":fedot,"terra_single_agent":terra},"provider":os.getenv("SAMPO_PROVIDER","openai-compatible"),"endpoint":os.getenv("SAMPO_BASE_URL") or os.getenv("OPENAI_BASE_URL"),"pricing_entries":sorted(entries),"blocking_requirements":missing,"live_inference_ready":not missing}
        print(json.dumps(result,indent=2))
        if missing: raise SystemExit(2)
    elif args.command=="introspect":
        cheap=os.getenv("SAMPO_FEDOT_MODEL") or "<SAMPO_FEDOT_MODEL required>";strong=os.getenv("SAMPO_TERRA_MODEL") or "<SAMPO_TERRA_MODEL required>"
        rows=read_csv(OUT/"operational_inputs.csv")[:20];labels=[x["target_label"] for x in read_csv(OUT/"allowed_target_labels.csv")]
        m=asyncio.run(get_manifest(rows,labels,args.run_id,OUT/"operational_inputs.csv",cheap,strong))
        print(json.dumps(m["prompts"],ensure_ascii=False,indent=2));print(json.dumps(m["mcp_tools"],ensure_ascii=False,indent=2))
    else:
        asyncio.run(execute(args.run_id,"smoke"))
if __name__=="__main__":main()
