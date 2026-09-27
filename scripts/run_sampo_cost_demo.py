#!/usr/bin/env python3
"""Execute cost-demo systems on public IDs only."""
from __future__ import annotations
import argparse, asyncio, csv, hashlib, json, os, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sampo_baselines import tfidf_char_ngrams
from sampo_cost_demo import OUT, ROOT, read_csv
from sampo_cost_runtime import BUDGETS, FEDOT_BUDGETS, NEUTRAL_TASK, NEUTRAL_SYSTEM, RuntimeTrace, introspect, model_costs, pricing_preflight, required_runtime, scoped_server, write_system_config

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
    ids=[r["example_id"] for r in rows]; out=run_root/system/f"batch_{batch:04d}"
    budgets=FEDOT_BUDGETS if phase5 else BUDGETS
    if system == "terra_single_agent":
        from sampo_terra_standalone import run as run_standalone_terra
        return await run_standalone_terra(
            model=model, endpoint=endpoint,
            api_key=os.getenv("SAMPO_API_KEY") or os.getenv("OPENAI_API_KEY", ""),
            rows=rows, run_id=run_id, output_dir=out, public_file=public_file,
            labels_file=OUT/"allowed_target_labels.csv", system_prompt=NEUTRAL_SYSTEM,
            user_prompt=neutral_user(rows,run_id), budgets=budgets, git_commit=commit(),
        )
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
    out.mkdir(parents=True,exist_ok=False)
    os.environ["FEDOTMAS_LOG_DIR"] = str((out / "logs").resolve())
    registry=scoped_server(ids,run_id,out,public_file,phase5=phase5,batch=batch)
    server_name="sampo-cost-demo-phase5" if phase5 else "sampo-cost-demo"
    description,tools=await introspect(registry,server_name);trace=RuntimeTrace(ids,run_id,out,model,{x["name"] for x in tools},batch=batch,allowed_labels=set(labels))
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
            await run_pipeline(app,user,session_service=InMemorySessionService(),timeout=budgets["max_batch_seconds"])
        except Exception as exc: trace.failures.append(f"execution_error:{type(exc).__name__}:{str(exc)[:300]}")
    else:
        toolset=create_toolset(server_name,registry=registry);instruction=NEUTRAL_SYSTEM
        user=neutral_user(rows,run_id)
        prompt_record={"system_instruction":instruction,"user_message":user}
        llm=make_llm(model_config)
        if not isinstance(getattr(llm,"llm_client",None),_ProxyClient): raise RuntimeError("Configured model did not resolve to observable _ProxyClient")
        app=App(name="sampo_cost_demo_agent",root_agent=LlmAgent(name="sampo_cost_demo_agent",model=llm,instruction=instruction,tools=[toolset]),plugins=[trace.plugin])
        started=time.monotonic()
        try: await run_pipeline(app,user,session_service=InMemorySessionService(),timeout=budgets["max_batch_seconds"])
        except Exception as exc: trace.failures.append(f"execution_error:{type(exc).__name__}:{str(exc)[:300]}")
    duration=time.monotonic()-started;rows_out=runtime_rows(out)
    with (out/"predictions.csv").open("x",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=["example_id","top_1","top_2","top_3"]);writer.writeheader();writer.writerows(rows_out)
    _ProxyClient.request_observer=None
    telemetry=trace.dump();telemetry.update({"run_id":run_id,"model":model,"completed_ids":[r["example_id"] for r in rows_out],"failed_ids":[i for i in ids if i not in {r["example_id"] for r in rows_out}],"completed_examples":len(rows_out),"model_calls_count":len(trace.calls),"mcp_calls_count":len(trace.tools),"runtime_seconds":duration,"safety_ceiling_hit":bool(trace.failures and any("limit" in f for f in trace.failures))})
    pricing=json.loads((OUT/"pricing.json").read_text());telemetry["cost_usd"]=model_costs(telemetry,pricing);telemetry["cost_complete"]=telemetry["cost_usd"] is not None
    runtime={"run_id":run_id,"system":system,"model":model,"provider":os.getenv("SAMPO_PROVIDER","openai-compatible"),"endpoint":endpoint,"model_configuration_path":"ModelConfig(api_base=required_runtime endpoint, api_key=SAMPO_API_KEY or OPENAI_API_KEY) -> make_llm -> _ProxyClient","batch_size":len(ids),"budgets":budgets,"fresh_session":True,"prompts":prompt_record,"tool_server_description":description,"tools":tools,"private_gt_path_provided":False,"phase5_config_sha256":digest(PHASE5_CONFIG) if phase5 else None,"git_commit":commit()}
    transcript=[{"type":"exposed_prompt_and_tools","system_instruction":prompt_record["system_instruction"],"user_message":prompt_record["user_message"],"server_description":description,"tools":tools},*trace.transcript]
    write_system_config(out,system,runtime,telemetry,transcript)
    return telemetry

async def get_manifest(rows: list[dict[str,str]], labels: list[str], run_id: str, public_file: Path, fedot: str, terra: str, run_stage: str = "smoke") -> dict:
    ids=[r["example_id"] for r in rows]
    probe=OUT/".schema_probe";probe.mkdir(parents=True,exist_ok=True)
    neutral=scoped_server(ids,run_id,probe,public_file);phase=scoped_server(ids,run_id,probe,public_file,phase5=True)
    _,neutral_tools=await introspect(neutral,"sampo-cost-demo");_,phase_tools=await introspect(phase,"sampo-cost-demo-phase5")
    user=neutral_user(rows,run_id)
    phase_config=json.loads(PHASE5_CONFIG.read_text(encoding="utf-8"))
    phase_user=_phase5_user(rows,run_id,0)
    final_input=(OUT/"public_inputs.csv").resolve()
    allowed_final=run_stage=="final" and public_file.resolve()==final_input
    return {"run_id":run_id,"run_stage":run_stage,"input_file":public_file.resolve().relative_to(ROOT).as_posix(),"input_sha256":digest(public_file),"assigned_ids":ids,"private_gt_evaluation_allowed":allowed_final,"git_commit":commit(),"created_at_utc":datetime.now(timezone.utc).isoformat(),"frozen_test_sha256":digest(OUT/"public_inputs.csv"),"operational_set_sha256":digest(OUT/"operational_inputs.csv"),"models":{"tfidf":None,"fedotmas_cost_aware":fedot,"terra_single_agent":terra},"provider":os.getenv("SAMPO_PROVIDER","openai-compatible"),"endpoint":os.getenv("SAMPO_BASE_URL") or os.getenv("OPENAI_BASE_URL"),"batch_size":20,"budgets":BUDGETS,"primary_question":"Can a FEDOT.MAS cost-aware workflow using DeepSeek v4.1 flash achieve accuracy reasonably close to a strong GPT-5.6 Terra single agent while using substantially less inference cost?","prompts":{"terra_single_agent":{"system":NEUTRAL_SYSTEM,"user":user},"fedotmas_cost_aware":{"coordinator_instruction":phase_config["coordinator"]["instruction"],"worker_instructions":[w["instruction"] for w in phase_config["workers"]],"user":phase_user,"saved_phase5_task":PHASE5_TASK,"config_sha256":digest(PHASE5_CONFIG)},"one_time_config_generation":{"usd":0,"basis":"Reuse of frozen saved Phase 5 config; no meta-generation during this experiment"}},"mcp_tools":{"neutral":neutral_tools,"phase5_adapter_only_for_fedotmas":phase_tools},"pricing_sha256":digest(OUT/"pricing.json"),"phase5_config_sha256":digest(PHASE5_CONFIG),"private_gt_exposed":False}

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

def smoke_system_failed(info: dict[str,Any], phase5: bool=False) -> bool:
    failures=info.get("failures",[])
    assigned=len(info.get("assigned_ids",[])) or 20
    if phase5 and info.get("terminal_success") and info.get("completed_examples")==assigned:
        failures=[failure for failure in failures if "duplicate_worker_delegation" not in str(failure)]
    return (info.get("completed_examples")!=assigned or not info.get("cost_complete") or info.get("output_truncation_hit",False) or info.get("safety_ceiling_hit",False) or bool(failures) or (phase5 and not info.get("terminal_success",False)))

def write_metrics(path: Path, run_id: str, info: dict[str,Any], harness: str) -> dict[str,Any]:
    assigned=info.get("assigned_ids",[]); completed=info.get("completed_ids",[])
    metrics={"run_id":run_id,"stage":info.get("run_stage"),"system":info.get("system"),"model":info.get("model"),"harness":harness,
        "assigned_examples":len(assigned),"completed_examples":len(completed),"failed_examples":max(0,len(assigned)-len(completed)),
        "provider_request_count":info.get("provider_request_count",0),"model_call_count":info.get("model_calls_count",0),
        "MCP_call_count":info.get("mcp_calls_count",len(info.get("mcp_calls",[]))),
        "uncached_input_tokens":info.get("uncached_input_tokens",0),"cached_input_tokens":info.get("cached_input_tokens",0),
        "total_input_tokens":info.get("input_tokens",0),"output_tokens":info.get("output_tokens",0),
        "provider_reported_cost_usd":info.get("provider_reported_cost_usd"),
        "fallback_calculated_cost_usd":info.get("fallback_calculated_cost_usd"),
        "authoritative_total_cost_usd":info.get("cost_usd"),"cost_complete":bool(info.get("cost_complete")),
        "runtime_seconds":info.get("runtime_seconds",0),"finish_reasons":info.get("finish_reasons",[]),
        "output_truncation_count":info.get("output_truncation_count",0),
        "safety_ceiling_hit":bool(info.get("safety_ceiling_hit")),"operational_pass":not smoke_system_failed(info,info.get("system")=="fedotmas_cost_aware")}
    if info.get("system")=="fedotmas_cost_aware":
        partition=info.get("phase5_partition_metrics",{})
        metrics.update({"coordinator_model_calls":info.get("coordinator_model_calls",0),"worker_model_calls":info.get("worker_model_calls",0),
            "worker_delegation_attempts":info.get("worker_delegations",0),"accepted_worker_delegations":info.get("accepted_worker_delegations",0),
            "review_count":partition.get("review_count",0),"fallback_count":partition.get("fallback_count",0),
            "semantic_review_fraction":partition.get("semantic_review_fraction",0),"evidence_requested_ids":partition.get("evidence_requested_ids",[]),
            "terminal_reason":info.get("terminal_reason")})
    path.write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+"\n")
    return metrics

def _aggregate_system(system: str, model: str, run_id: str, run_root: Path, all_rows: list[dict[str,str]], batches: list[dict[str,Any]], experiment: dict[str,Any], phase5: bool) -> dict:
    system_dir=run_root/system;system_dir.mkdir(exist_ok=True)
    predictions=[];calls=[];provider_requests=[];mcp_calls=[];worker_calls=[];failures=[];transcript=[];per_batch=[];elapsed=0.0;delegation_attempts=0;terminal_success=False;terminal_reason=None
    for index,batch_rows in enumerate(batches):
        bdir=system_dir/f"batch_{index:04d}"
        if not bdir.exists():
            failures.append({"batch":index,"error":"batch_artifacts_missing"});continue
        predictions.extend(read_csv(bdir/"predictions.csv"))
        telemetry=json.loads((bdir/"telemetry.json").read_text())
        calls.extend(telemetry.get("model_calls",[]));provider_requests.extend(telemetry.get("provider_requests",[]));mcp_calls.extend(telemetry.get("mcp_calls",[]));worker_calls.extend(telemetry.get("worker_calls",[]));delegation_attempts+=telemetry.get("worker_delegations",0);terminal_success=terminal_success or telemetry.get("terminal_success",False);terminal_reason=telemetry.get("terminal_reason") or terminal_reason;failures.extend({"batch":index,"error":x} for x in telemetry.get("failures",[]));elapsed+=telemetry.get("runtime_seconds",0)
        transcript.extend((bdir/"transcript.jsonl").read_text().splitlines())
        per_batch.append(json.loads((bdir/"runtime_manifest.json").read_text()))
    attempts_dir=system_dir/".attempts"
    if attempts_dir.exists():
        for attempt_path in sorted(attempts_dir.glob("batch_*/telemetry.json")):
            try: prior=json.loads(attempt_path.read_text())
            except (OSError,json.JSONDecodeError): continue
            calls.extend(prior.get("model_calls",[]));provider_requests.extend(prior.get("provider_requests",[]))
            mcp_calls.extend(prior.get("mcp_calls",[]));worker_calls.extend(prior.get("worker_calls",[]))
            delegation_attempts+=prior.get("worker_delegations",0);elapsed+=prior.get("runtime_seconds",0)
    for call in mcp_calls:
        response=call.get("response")
        if isinstance(response,dict) and response.get("isError") in (True,"True"):
            messages=[item.get("text","") for item in response.get("content",[]) if isinstance(item,dict)]
            failures.append({"tool":call.get("name"),"error":"; ".join(messages)[:500] or "MCP tool returned isError"})
    order={r["example_id"]:i for i,r in enumerate(all_rows)}
    predictions.sort(key=lambda r:order[r["example_id"]])
    pred_path=system_dir/"predictions.csv"
    with pred_path.open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=["example_id","top_1","top_2","top_3"]);writer.writeheader();writer.writerows(predictions)
    completed={r["example_id"] for r in predictions};ids=[r["example_id"] for r in all_rows]
    pricing=json.loads((OUT/"pricing.json").read_text());cost=model_costs({"model_calls":calls,"model_calls_count":len(calls),"provider_requests":provider_requests,"proxy_observable":True},pricing)
    known=[x for x in provider_requests if x["usage_known"]]
    truncations=sum(c.get("finish_reason") in {"length","max_tokens"} for c in provider_requests)
    reported_costs=[float(c["provider_cost_usd"]) for c in provider_requests if c.get("provider_cost_usd") is not None]
    provider_cost=sum(reported_costs) if reported_costs else None
    fallback_cost=sum(call_cost(c,pricing) for c in provider_requests if c.get("usage_known") and c.get("provider_cost_usd") is None)
    telemetry={"run_id":run_id,"run_stage":experiment.get("run_stage"),"system":system,"model":model,"assigned_ids":ids,"completed_ids":[i for i in ids if i in completed],"failed_ids":[i for i in ids if i not in completed],"completed_examples":len(completed),"model_calls":calls,"provider_requests":provider_requests,"provider_request_count":len(provider_requests),"malformed_tool_retry_requests":sum(x["request_kind"]=="malformed_tool_retry" for x in provider_requests),"proxy_observable":True,"cost_complete":cost is not None,"mcp_calls":mcp_calls,"worker_calls":worker_calls,"worker_delegations":delegation_attempts,"accepted_worker_delegations":sum(c.get("name","").endswith("construction_batch_specialist") for c in worker_calls),"terminal_success":terminal_success,"terminal_reason":terminal_reason,"tool_calls":len(mcp_calls),"input_tokens":sum(c["input_tokens"] for c in known),"uncached_input_tokens":sum(max(0,c["input_tokens"]-c["cached_input_tokens"]) for c in known),"cached_input_tokens":sum(c["cached_input_tokens"] for c in known),"output_tokens":sum(c["output_tokens"] for c in known),"provider_reported_cost_usd":provider_cost,"fallback_calculated_cost_usd":fallback_cost,"finish_reasons":[c.get("finish_reason") for c in provider_requests],"output_truncation_count":truncations,"output_truncation_hit":truncations>0,"adk_input_tokens":sum(c["input_tokens"] for c in calls),"adk_cached_input_tokens":sum(c["cached_input_tokens"] for c in calls),"adk_output_tokens":sum(c["output_tokens"] for c in calls),"model_calls_count":len(calls) if phase5 else len(provider_requests),"coordinator_model_calls":sum(c.get("agent_name")=="construction_label_batch_coordinator" for c in calls),"worker_model_calls":sum(c.get("agent_name")=="construction_batch_specialist" for c in calls),"mcp_calls_count":len(mcp_calls),"cost_usd":cost,"runtime_seconds":elapsed,"safety_ceiling_hit":any("limit" in str(x) for x in failures),"failures":failures}
    if phase5:
        telemetry["phase5_partition_metrics"]=phase5_partition_metrics({"mcp_calls":mcp_calls},len(ids))
    telemetry["operational_pass"]=not smoke_system_failed(telemetry,phase5)
    runtime={"run_id":run_id,"system":system,"harness":"FEDOT.MAS" if phase5 else "standalone_openai_tool_loop","fedotmas_dependency":bool(phase5),"model":model,"batch_size":20,"budgets":BUDGETS,"fresh_session_per_batch":True,"batches":per_batch,"phase5_frozen_config_sha256":digest(PHASE5_CONFIG) if phase5 else None,"git_commit":experiment["git_commit"],"private_ground_truth_path_provided":False,"fedotmas_predictions_exposed_to_terra":False,"phase5_config_exposed_to_terra":False}
    (system_dir/"telemetry.json").write_text(json.dumps(telemetry,ensure_ascii=False,indent=2)+"\n")
    (system_dir/"runtime_manifest.json").write_text(json.dumps(runtime,ensure_ascii=False,indent=2)+"\n")
    with (system_dir/"transcript.jsonl").open("w",encoding="utf-8") as f:
        for line in transcript: f.write(line+"\n")
    write_metrics(system_dir/"metrics.json",run_id,telemetry,"FEDOT.MAS" if phase5 else "standalone_openai_tool_loop")
    return telemetry

STAGES = {"smoke": (OUT / "operational_inputs.csv", 20),
          "operational": (OUT / "operational_200_inputs.csv", 200),
          "final": (OUT / "public_inputs.csv", 1000)}
SYSTEMS = ("tfidf", "fedotmas_cost_aware", "terra_single_agent")


def make_run_id(stage: str) -> str:
    count = STAGES[stage][1]
    base=f"sampo-{stage}-{count}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    candidate=base; suffix=2
    while (OUT/"runs"/candidate).exists():
        candidate=f"{base}-{suffix}"; suffix+=1
    return candidate


def config_signature(stage: str, input_path: Path, fedot: str, terra: str, endpoint: str) -> dict:
    return {"run_stage": stage, "input_file": input_path.resolve().relative_to(ROOT).as_posix(),
            "input_sha256": digest(input_path), "models": {"tfidf": "tfidf_char_ngrams",
            "fedotmas_cost_aware": fedot, "terra_single_agent": terra},
            "endpoint": endpoint, "provider": os.getenv("SAMPO_PROVIDER", "openai-compatible"),
            "phase5_config_sha256": digest(PHASE5_CONFIG), "pricing_sha256": digest(OUT/"pricing.json"),
            "batch_size": 20, "budgets": {"fedotmas_cost_aware": FEDOT_BUDGETS,
                                             "terra_single_agent": BUDGETS}}


def write_batch_failure(path: Path, system: str, model: str, run_id: str, rows: list[dict], error: Exception, phase5: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    ids = [r["example_id"] for r in rows]
    info = {"run_id": run_id, "system": system, "model": model, "assigned_ids": ids,
            "completed_ids": [], "failed_ids": ids, "completed_examples": 0,
            "provider_requests": [], "provider_request_count": 0, "model_calls": [],
            "model_calls_count": 0, "mcp_calls": [], "mcp_calls_count": 0,
            "worker_calls": [], "worker_delegations": 0, "input_tokens": 0,
            "uncached_input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0,
            "provider_reported_cost_usd": None, "fallback_calculated_cost_usd": None,
            "cost_usd": None, "cost_complete": False, "runtime_seconds": 0,
            "finish_reasons": [], "output_truncation_count": 0, "safety_ceiling_hit": False,
            "failures": [f"batch_error:{type(error).__name__}:{str(error)[:400]}"], "proxy_observable": True}
    with (path/"predictions.csv").open("w", encoding="utf-8", newline="") as f:
        csv.DictWriter(f, fieldnames=["example_id","top_1","top_2","top_3"]).writeheader()
    (path/"telemetry.json").write_text(json.dumps(info, indent=2)+"\n")
    (path/"runtime_manifest.json").write_text(json.dumps({"run_id":run_id,"system":system,"model":model,
        "batch_size":len(rows),"fresh_session":True,"private_gt_path_provided":False,
        "fedotmas_dependency":phase5,"error":info["failures"][0]}, indent=2)+"\n")
    (path/"transcript.jsonl").write_text("")


def batch_complete(path: Path, ids: list[str], run_id: str, system: str) -> bool:
    try:
        telemetry=json.loads((path/"telemetry.json").read_text())
        runtime=json.loads((path/"runtime_manifest.json").read_text())
        predictions=read_csv(path/"predictions.csv")
        done={r["example_id"] for r in predictions}
        failures=telemetry.get("failures",[])
        if system=="fedotmas_cost_aware" and telemetry.get("terminal_success") and telemetry.get("completed_examples")==len(ids):
            failures=[failure for failure in failures if "duplicate_worker_delegation" not in str(failure)]
        return (telemetry.get("run_id")==run_id and runtime.get("run_id")==run_id and
                telemetry.get("system")==system and telemetry.get("failed_ids")==[] and
                done==set(ids) and telemetry.get("completed_examples")==len(ids) and
                not failures and not telemetry.get("output_truncation_hit",False) and
                not telemetry.get("safety_ceiling_hit",False) and
                telemetry.get("cost_complete",system=="tfidf"))
    except (OSError, ValueError, json.JSONDecodeError, KeyError):
        return False


async def execute(run_id: str, stage: str, resume: bool = False) -> int:
    if stage == "operational" and not (OUT/"operational_200_inputs.csv").exists():
        from sampo_cost_demo import construct_operational_200
        construct_operational_200()
    fedot, terra, _, endpoint = required_runtime()
    pricing_preflight({fedot, terra})
    public_file, expected = STAGES[stage]
    rows=read_csv(public_file)
    if len(rows)!=expected: raise RuntimeError(f"{stage} input must contain exactly {expected} examples; found {len(rows)}")
    labels=[x["target_label"] for x in read_csv(OUT/"allowed_target_labels.csv")]
    batches=[rows[i:i+20] for i in range(0,len(rows),20)]
    signature=config_signature(stage,public_file,fedot,terra,endpoint)
    run_root=OUT/"runs"/run_id
    if resume:
        manifest_path=run_root/"experiment_manifest.json"
        if not manifest_path.is_file(): raise RuntimeError("Cannot resume: experiment manifest is missing")
        manifest=json.loads(manifest_path.read_text())
        if (manifest.get("run_id")!=run_id or manifest.get("configuration_signature")!=signature or
                manifest.get("run_stage")!=stage or manifest.get("private_gt_evaluation_allowed") is not (stage=="final") or
                manifest.get("assigned_ids")!=[r["example_id"] for r in rows]):
            raise RuntimeError("Cannot resume: run manifest configuration or input hash differs")
    else:
        run_root.mkdir(parents=True,exist_ok=False)
        manifest={**signature,"run_id":run_id,"input_file":signature["input_file"],"input_sha256":signature["input_sha256"],
                  "assigned_ids":[r["example_id"] for r in rows],"private_gt_evaluation_allowed":stage=="final",
                  "created_at_utc":datetime.now(timezone.utc).isoformat(),"git_commit":commit(),
                  "frozen_test_sha256":digest(OUT/"public_inputs.csv"),"operational_set_sha256":digest(OUT/"operational_inputs.csv") if (OUT/"operational_inputs.csv").exists() else None,
                  "configuration_signature":signature,"private_gt_exposed":False}
        (run_root/"experiment_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
    print(f"\n=== SAMPO {stage.upper()} RUN: {run_id} ===", flush=True)
    any_failed=False; infos={}
    for system,model,phase5 in (("tfidf","tfidf_char_ngrams",False),("fedotmas_cost_aware",fedot,True),("terra_single_agent",terra,False)):
        for idx, chunk in enumerate(batches):
            bdir=run_root/system/f"batch_{idx:04d}"
            if system=="tfidf":
                if batch_complete(bdir,[r["example_id"] for r in chunk],run_id,system): continue
                bdir.mkdir(parents=True,exist_ok=True)
                pr=tfidf_batch(chunk,labels); t={"run_id":run_id,"run_stage":stage,"system":system,"model":model,"assigned_ids":[r["example_id"] for r in chunk],"completed_ids":[r["example_id"] for r in pr],"failed_ids":[],"completed_examples":len(pr),"cost_usd":0.0,"cost_complete":True,"provider_request_count":0,"model_calls_count":0,"mcp_calls_count":0,"input_tokens":0,"uncached_input_tokens":0,"cached_input_tokens":0,"output_tokens":0,"provider_reported_cost_usd":0.0,"fallback_calculated_cost_usd":0.0,"runtime_seconds":0,"failures":[]}
                with (bdir/"predictions.csv").open("w",encoding="utf-8",newline="") as f:
                    w=csv.DictWriter(f,fieldnames=["example_id","top_1","top_2","top_3"]);w.writeheader();w.writerows(pr)
                (bdir/"telemetry.json").write_text(json.dumps(t,indent=2)+"\n")
                (bdir/"runtime_manifest.json").write_text(json.dumps({"run_id":run_id,"system":system,"model":model,"batch_size":len(chunk)})+"\n")
                (bdir/"transcript.jsonl").write_text("")
                continue
            if batch_complete(bdir,[r["example_id"] for r in chunk],run_id,system): continue
            try:
                (run_root/system).mkdir(parents=True,exist_ok=True)
                if bdir.exists():
                    attempts_dir=run_root/system/".attempts"; attempts_dir.mkdir(exist_ok=True)
                    attempt=1
                    while (attempts_dir/f"batch_{idx:04d}_attempt_{attempt:03d}").exists(): attempt+=1
                    bdir.rename(attempts_dir/f"batch_{idx:04d}_attempt_{attempt:03d}")
                await run_agent(system,model,endpoint,chunk,labels,run_id,run_root,public_file,phase5,batch=idx)
            except Exception as exc:
                write_batch_failure(bdir,system,model,run_id,chunk,exc,phase5)
        try:
            info=_aggregate_system(system,model,run_id,run_root,rows,batches,manifest,phase5)
        except Exception as exc:
            any_failed=True
            info={"run_id":run_id,"system":system,"assigned_ids":[r["example_id"] for r in rows],"completed_ids":[],"completed_examples":0,"cost_complete":False,"cost_usd":None,"failures":[str(exc)]}
        infos[system]=info
        if info.get("completed_examples")!=expected or smoke_system_failed(info,system=="fedotmas_cost_aware"): any_failed=True
        print(f"{system}: {info.get('completed_examples',0)}/{expected}; cost={info.get('cost_usd')}; cost_complete={info.get('cost_complete')}",flush=True)
        if stage=="smoke" and system=="fedotmas_cost_aware" and info.get("output_truncation_count",0)>0:
            print("STOP: FEDOT.MAS provider output was truncated at the configured output limit; Terra smoke was not started.",flush=True)
            break
    ft=infos.get("fedotmas_cost_aware",{}); tt=infos.get("terra_single_agent",{})
    ratio=tt.get("cost_usd")/ft.get("cost_usd") if tt.get("cost_usd") is not None and ft.get("cost_usd") else None
    root={"run_id":run_id,"stage":stage,"completion_counts":{s:{"completed":x.get("completed_examples",0),"assigned":expected} for s,x in infos.items()},
          "operational_pass":{s:(x.get("completed_examples")==expected and not x.get("failures",[]) and x.get("cost_complete",False)) for s,x in infos.items()},
          "fedotmas_total_cost_usd":ft.get("cost_usd"),"terra_total_cost_usd":tt.get("cost_usd"),"terra_over_fedotmas_cost_ratio":ratio,
          "token_ratios":{"input":tt.get("input_tokens",0)/ft.get("input_tokens") if ft.get("input_tokens") else None,"output":tt.get("output_tokens",0)/ft.get("output_tokens") if ft.get("output_tokens") else None,"total":(tt.get("input_tokens",0)+tt.get("output_tokens",0))/(ft.get("input_tokens",0)+ft.get("output_tokens",0)) if ft.get("input_tokens",0)+ft.get("output_tokens",0) else None},
          "provider_request_ratio":tt.get("provider_request_count",0)/ft.get("provider_request_count") if ft.get("provider_request_count") else None,
          "runtime_ratio":tt.get("runtime_seconds",0)/ft.get("runtime_seconds") if ft.get("runtime_seconds") else None,
          "cost_complete":{"fedotmas_cost_aware":ft.get("cost_complete",False),"terra_single_agent":tt.get("cost_complete",False)},"private_gt_used":False}
    (run_root/"metrics_gt_blind.json").write_text(json.dumps(root,ensure_ascii=False,indent=2)+"\n")
    print(f"\nCompleted run ID: {run_id}",flush=True)
    if stage=="final" and not any_failed:
        print("Final inference complete.\n\nRun ID:\n"+run_id+"\n\nEvaluate with:\n.venv/bin/python scripts/evaluate_sampo_cost_demo.py --run-id "+run_id,flush=True)
    return 1 if any_failed else 0


def status(run_id: str) -> int:
    root=OUT/"runs"/run_id; mp=root/"experiment_manifest.json"
    if not mp.is_file(): raise SystemExit(f"Unknown run ID: {run_id}")
    manifest=json.loads(mp.read_text()); stage=manifest.get("run_stage"); assigned=len(manifest.get("assigned_ids",[]))
    print(f"Run: {run_id}\nStage: {stage}\n")
    complete=True
    for system in SYSTEMS:
        path=root/system/"metrics.json"
        m=json.loads(path.read_text()) if path.exists() else {}
        done=m.get("completed_examples",0); cost=m.get("authoritative_total_cost_usd")
        print(f"{system:14} {done}/{assigned}"+(f"   ${cost:.4f}" if isinstance(cost,(int,float)) else ""))
        complete &= done==assigned and bool(m.get("cost_complete",system=="tfidf"))
    print(f"\nCost telemetry complete: {'yes' if complete else 'no'}")
    evaluation=root/"evaluation.json"
    print("Ready for evaluation: "+("yes" if stage=="final" and complete else "no"))
    if evaluation.exists():
        e=json.loads(evaluation.read_text()); c=json.loads((root/"comparison.json").read_text())
        print(f"\nAccuracy: FEDOT.MAS {e['systems']['fedotmas_cost_aware']['top1_accuracy']:.3%}; Terra {e['systems']['terra_single_agent']['top1_accuracy']:.3%}")
        print(f"Primary criterion: {'pass' if c['overall_primary_pass'] else 'fail'}")
    else:
        print("\nEvaluation: not run")
        if stage=="final" and complete: print(f"\nCommand:\npython scripts/evaluate_sampo_cost_demo.py --run-id {run_id}")
    return 0


def main() -> None:
    parser=argparse.ArgumentParser(); sub=parser.add_subparsers(dest="command",required=True)
    sub.add_parser("preflight")
    intro=sub.add_parser("introspect"); intro.add_argument("--run-id",default="schema-inspection")
    for stage in ("smoke","operational","final"):
        cmd=sub.add_parser(stage); cmd.add_argument("--run-id"); cmd.add_argument("--resume",action="store_true")
    stat=sub.add_parser("status"); stat.add_argument("--run-id",required=True)
    args=parser.parse_args()
    if args.command=="preflight":
        fedot,terra=os.getenv("SAMPO_FEDOT_MODEL"),os.getenv("SAMPO_TERRA_MODEL")
        pricing=json.loads((OUT/"pricing.json").read_text()); entries=pricing.get("models",{})
        missing=[]
        if fedot!="deepseek/deepseek-v4.1-flash": missing.append("SAMPO_FEDOT_MODEL=deepseek/deepseek-v4.1-flash")
        if terra!="openai/gpt-5.6-terra": missing.append("SAMPO_TERRA_MODEL=openai/gpt-5.6-terra")
        if not (os.getenv("SAMPO_BASE_URL") or os.getenv("OPENAI_BASE_URL")): missing.append("provider endpoint")
        if not (os.getenv("SAMPO_API_KEY") or os.getenv("OPENAI_API_KEY")): missing.append("SAMPO_API_KEY/OPENAI_API_KEY")
        for model in (fedot,terra):
            if model and model not in entries: missing.append(f"pricing entry for {model}")
        if not missing:
            try: pricing_preflight({fedot,terra})
            except Exception as exc: missing.append(str(exc))
        result={"models":{"fedotmas":fedot,"terra_single_agent":terra},"provider":os.getenv("SAMPO_PROVIDER","openai-compatible"),"endpoint":os.getenv("SAMPO_BASE_URL") or os.getenv("OPENAI_BASE_URL"),"pricing_entries":sorted(entries),"blocking_requirements":missing,"live_inference_ready":not missing}
        print(json.dumps(result,indent=2)); raise SystemExit(2 if missing else 0)
    if args.command=="status": raise SystemExit(status(args.run_id))
    if args.command=="introspect":
        cheap=os.getenv("SAMPO_FEDOT_MODEL") or "deepseek/deepseek-v4.1-flash"; strong=os.getenv("SAMPO_TERRA_MODEL") or "openai/gpt-5.6-terra"
        rows=read_csv(OUT/"operational_inputs.csv")[:20]; labels=[x["target_label"] for x in read_csv(OUT/"allowed_target_labels.csv")]
        m=asyncio.run(get_manifest(rows,labels,args.run_id,OUT/"operational_inputs.csv",cheap,strong)); print(json.dumps(m["mcp_tools"],ensure_ascii=False,indent=2)); return
    run_id=args.run_id or make_run_id(args.command)
    print(f"Run ID: {run_id}",flush=True)
    raise SystemExit(asyncio.run(execute(run_id,args.command,args.resume)))

if __name__=="__main__": main()
