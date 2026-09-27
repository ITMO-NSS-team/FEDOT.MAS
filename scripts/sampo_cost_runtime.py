"""Shared GT-blind execution, tracing, schema introspection, and preflight."""
from __future__ import annotations
import asyncio, csv, json, os, re, shutil, subprocess, sys, time
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from dotenv import find_dotenv, load_dotenv
from sampo_cost_demo import OUT, ROOT, read_csv, call_cost

_DOTENV_PATH=find_dotenv(usecwd=True)
if _DOTENV_PATH: load_dotenv(_DOTENV_PATH,override=False)

BUDGETS = {"batch_size": 20, "max_model_calls": 30, "max_mcp_calls": 60, "max_prompt_tokens_per_call": 64000, "max_batch_seconds": 300, "max_output_tokens_per_call": 8192}
FEDOT_BUDGETS = {**BUDGETS, "max_output_tokens_per_call": 16384}
NEUTRAL_TASK = "Map each assigned historical construction work name to three distinct allowed labels using only the supplied public data and tools. Produce valid top-3 predictions for all assigned IDs."
NEUTRAL_SYSTEM = "You are a single agent completing a batch of SAMPO construction work name mappings. Follow the task and use only public batch data, allowed labels, and available tools. Return valid top-three predictions for every assigned ID."

def required_runtime() -> tuple[str, str, str, str]:
    cheap, strong = os.getenv("SAMPO_FEDOT_MODEL"), os.getenv("SAMPO_TERRA_MODEL")
    endpoint = os.getenv("SAMPO_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    provider = os.getenv("SAMPO_PROVIDER", "openai-compatible" if endpoint else "")
    missing = [name for name, value in (("SAMPO_FEDOT_MODEL", cheap), ("SAMPO_TERRA_MODEL", strong), ("SAMPO_PROVIDER/endpoint", provider and endpoint)) if not value]
    if cheap and cheap != "deepseek/deepseek-v4.1-flash": missing.append("SAMPO_FEDOT_MODEL must be deepseek/deepseek-v4.1-flash")
    if strong and strong != "openai/gpt-5.6-terra": missing.append("SAMPO_TERRA_MODEL must be openai/gpt-5.6-terra")
    if not (os.getenv("SAMPO_API_KEY") or os.getenv("OPENAI_API_KEY")): missing.append("SAMPO_API_KEY/OPENAI_API_KEY")
    if missing: raise RuntimeError("Missing live inference configuration: " + ", ".join(missing))
    return cheap, strong, provider, endpoint

def model_config_for(model: str, endpoint: str):
    """Build an explicit OpenAI-compatible model config; never use LiteLLM fallback."""
    from fedotmas._settings import ModelConfig
    api_key=os.getenv("SAMPO_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key: raise RuntimeError("Missing SAMPO_API_KEY or OPENAI_API_KEY")
    return ModelConfig(model=model,api_base=endpoint,api_key=api_key)

def pricing_preflight(models: set[str]) -> dict[str, Any]:
    path = OUT / "pricing.json"
    pricing = json.loads(path.read_text(encoding="utf-8"))
    entries = pricing.get("models", {})
    missing = models - entries.keys()
    if missing: raise RuntimeError("Pricing entries missing for: " + ", ".join(sorted(missing)) + "; explicit dated pricing is required before inference")
    for model in models:
        row = entries[model]
        for k in ("input_usd_per_1m", "output_usd_per_1m"):
            if isinstance(row.get(k), bool) or not isinstance(row.get(k), (int, float)) or not math.isfinite(row[k]) or row[k] < 0: raise RuntimeError(f"Invalid pricing field {model}.{k}")
        if not row.get("source") or not row.get("source_date"): raise RuntimeError(f"Pricing provenance missing for {model}")
    return pricing

def scoped_server(ids: list[str], run_id: str, run_dir: Path, public_file: Path, phase5: bool = False, batch: int = 0):
    from fedotmas.mcp import StdioMCPServer
    package = ROOT / "mcp-servers/sampo-cost-demo"
    python = ROOT / "mcp-servers/sampo-phase6/.venv/bin/python"
    script = package / "src/mcp_sampo_cost_demo/server_phase5.py" if phase5 else package / "src/mcp_sampo_cost_demo/server.py"
    clean_env = {k: v for k, v in os.environ.items() if k in {"PATH", "TMPDIR", "TEMP", "LANG", "LC_ALL", "VIRTUAL_ENV"}}
    clean_env.update({"SAMPO_PUBLIC_INPUTS": str(public_file.resolve()), "SAMPO_ALLOWED_LABELS": str((OUT / "allowed_target_labels.csv").resolve()), "SAMPO_ASSIGNED_IDS": ",".join(ids), "SAMPO_RUN_ID": run_id, "SAMPO_RUN_DIR": str(run_dir.resolve()), "SAMPO_BATCH": str(batch)})
    name = "sampo-cost-demo-phase5" if phase5 else "sampo-cost-demo"
    return {name: StdioMCPServer(command=str(python), args=(str(script),), env=clean_env, description="Batch-scoped public SAMPO retrieval and prediction storage.")}

async def introspect(registry: dict[str, Any], server_name: str) -> tuple[str, list[dict[str, Any]]]:
    from mcp import StdioServerParameters
    from mcp.client.session import ClientSession
    from mcp.client.stdio import stdio_client
    cfg = registry[server_name]
    params = StdioServerParameters(command=cfg.command, args=list(cfg.args), env=cfg.env)
    async with stdio_client(params) as (rs, ws):
        async with ClientSession(rs, ws) as client:
            await client.initialize()
            result = await client.list_tools()
            tools = [x.model_dump(mode="json", exclude_none=True) for x in result.tools]
    return cfg.description, tools

def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)): return value
    if hasattr(value, "model_dump"): return value.model_dump(mode="json", exclude_none=True)
    if hasattr(value, "model_dump_json"): return json.loads(value.model_dump_json(exclude_none=True))
    if isinstance(value, Mapping): return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_jsonable(v) for v in value]
    return str(value)

class RuntimeTrace:
    """ADK plugin compatible callback object with hard pre-call ceilings."""
    def __init__(self, ids: list[str], run_id: str, output: Path, model: str, tool_names: set[str] | None = None, batch: int = 0, allowed_labels: set[str] | None = None):
        from google.adk.plugins import BasePlugin
        owner = self
        class Plugin(BasePlugin):
            async def before_model_callback(self, **kwargs): return await owner.before_model_callback(**kwargs)
            async def after_model_callback(self, **kwargs): return await owner.after_model_callback(**kwargs)
            async def before_tool_callback(self, **kwargs): return await owner.before_tool_callback(**kwargs)
            async def after_tool_callback(self, **kwargs): return await owner.after_tool_callback(**kwargs)
            async def on_tool_error_callback(self, **kwargs): return await owner.on_tool_error_callback(**kwargs)
        self.plugin = Plugin(name="sampo_cost_demo_trace")
        self.ids, self.run_id, self.output, self.model, self.batch = ids, run_id, output, model, batch
        self.max_output_tokens = (FEDOT_BUDGETS if output.parent.name == "fedotmas_cost_aware" else BUDGETS)["max_output_tokens_per_call"]
        self.tool_names=tool_names or set(); self.calls: list[dict[str, Any]]=[]; self.tools: list[dict[str, Any]]=[]; self.worker_calls: list[dict[str, Any]]=[]; self.provider_requests: list[dict[str, Any]]=[]; self.failures=[]; self.transcript=[]; self.started=time.monotonic(); self.call_started=0.0
        self.worker_delegations = 0
        self.allowed_labels=allowed_labels or set(); self.terminal_success=False; self.terminal_reason=None

    def record_provider_request(self, request: dict[str, Any]) -> None:
        observed={**request,"max_output_tokens":self.max_output_tokens}
        self.provider_requests.append(observed)
        self.transcript.append({"type": "provider_request", **observed})
        if observed.get("finish_reason") in {"length", "max_tokens"}:
            self.failures.append("output_truncation")

    async def before_model_callback(self, *, callback_context, llm_request):
        if self.terminal_success:
            invocation=getattr(callback_context,"_invocation_context",None)
            if invocation is not None: invocation.end_invocation=True
            from google.adk.models import LlmResponse
            from google.genai import types
            return LlmResponse(content=types.Content(role="model",parts=[types.Part(text="Assigned batch is durably complete.")]),turn_complete=True)
        config=getattr(llm_request,"config",None)
        if config is None:
            from google.genai import types
            llm_request.config=types.GenerateContentConfig(max_output_tokens=self.max_output_tokens)
        else:
            config.max_output_tokens=self.max_output_tokens
        if "duplicate_worker_delegation" in self.failures:
            raise RuntimeError("duplicate_worker_delegation")
        if len(self.calls) >= BUDGETS["max_model_calls"]: self.failures.append("model_call_limit"); raise RuntimeError("model_call_limit")
        payload = _jsonable(llm_request); encoded = json.dumps(payload, ensure_ascii=False)
        try:
            from litellm import token_counter
            estimate=int(token_counter(model=getattr(llm_request,"model",None) or self.model,text=encoded))
        except Exception:
            estimate=max(1,(len(encoded)+2)//3)
        if estimate > BUDGETS["max_prompt_tokens_per_call"]: self.failures.append("prompt_token_limit"); raise RuntimeError("prompt_token_limit")
        self.call_started=time.monotonic()
        self.transcript.append({"type":"model_request","agent":getattr(callback_context,"agent_name",None),"request":payload})

    async def after_model_callback(self, *, callback_context, llm_response):
        usage = llm_response.usage_metadata
        prompt = int((getattr(usage,"prompt_token_count",0) or 0)); output = int((getattr(usage,"candidates_token_count",0) or 0))
        cached = int((getattr(usage,"cached_content_token_count",0) or 0))
        row={"system":self.output.parent.name,"batch":self.batch,"example_ids":self.ids,"agent_name":getattr(callback_context,"agent_name",None),"model":self.model,"input_tokens":prompt,"cached_input_tokens":cached,"output_tokens":output,"elapsed_seconds":time.monotonic()-self.call_started,"provider_cost_usd":None}
        self.calls.append(row); self.transcript.append({"type":"model_response","usage":row,"response":_jsonable(llm_response)})

    async def before_tool_callback(self, *, tool, tool_args, tool_context):
        name=tool.name or "unknown"
        is_mcp=name in self.tool_names or any(name.endswith("_"+x) for x in self.tool_names)
        if self.terminal_success:
            self._terminate_invocation(tool_context)
            self.transcript.append({"type":"invocation_terminated","reason":self.terminal_reason,"before_tool":name})
            return {"status":"terminal_success","terminal_reason":self.terminal_reason}
        if is_mcp and len(self.tools)>=BUDGETS["max_mcp_calls"]: self.failures.append("mcp_call_limit"); raise RuntimeError("mcp_call_limit")
        args=tool_args or {}; ids=args.get("example_ids",[]) if isinstance(args, Mapping) else []
        if ids and not set(ids)<=set(self.ids): self.failures.append("out_of_scope_tool_ids"); raise RuntimeError("out_of_scope_tool_ids")
        if self.output.parent.name == "fedotmas_cost_aware" and name.endswith("construction_batch_specialist"):
            self.worker_delegations += 1
            if self.worker_delegations > 1:
                self.failures.append("duplicate_worker_delegation")
                raise RuntimeError("duplicate_worker_delegation")
        row={"name":name,"arguments":_jsonable(args)}
        (self.tools if is_mcp else self.worker_calls).append(row)
        self.transcript.append({"type":"tool_request","kind":"mcp" if is_mcp else "worker","name":name,"arguments":_jsonable(args)})

    async def after_tool_callback(self, *, tool, tool_args, tool_context, result):
        response=_jsonable(result)
        name=tool.name or "unknown"
        arguments=_jsonable(tool_args or {})
        rows=self.tools if name in self.tool_names or any(name.endswith("_"+x) for x in self.tool_names) else self.worker_calls
        for row in reversed(rows):
            if row["name"]==name and row.get("arguments")==arguments and "response" not in row:
                row["response"]=response
                break
        self.transcript.append({"type":"tool_response","name":name,"response":response})
        if self.terminal_success:
            self._terminate_invocation(tool_context)
            self.transcript.append({"type":"invocation_terminated","reason":self.terminal_reason,"after_tool":name})
            return
        persistence_tool=name.endswith(("save_review_decisions","save_candidate_predictions"))
        is_error=isinstance(response,Mapping) and response.get("isError") in (True,"True")
        if self.output.parent.name!="fedotmas_cost_aware" or not persistence_tool or is_error:
            return
        if self._durable_batch_complete():
            self.terminal_success=True
            self.terminal_reason="durable_batch_complete"
            self.transcript.append({"type":"terminal_success","reason":self.terminal_reason,"assigned_ids":self.ids})
            self._terminate_invocation(tool_context)

    @staticmethod
    def _terminate_invocation(tool_context: Any) -> None:
        actions=getattr(tool_context,"actions",None)
        if actions is not None:
            actions.end_of_agent=True
        invocation=getattr(tool_context,"_invocation_context",None)
        if invocation is None:
            raise RuntimeError("ADK tool context did not expose the invocation termination mechanism")
        invocation.end_invocation=True

    def _durable_batch_complete(self) -> bool:
        path=self.output/".predictions.jsonl"
        if not path.is_file() or not self.allowed_labels:
            return False
        try:
            rows=[json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        except (OSError,json.JSONDecodeError):
            return False
        ids=[row.get("example_id") for row in rows]
        if len(ids)!=len(self.ids) or len(set(ids))!=len(ids) or set(ids)!=set(self.ids):
            return False
        return all(
            len(values)==3 and len(set(values))==3 and set(values)<=self.allowed_labels
            for row in rows
            for values in [[row.get(f"top_{rank}") for rank in (1,2,3)]]
        )

    async def on_tool_error_callback(self, *, tool, tool_args, tool_context, error):
        self.failures.append(f"tool_error:{tool.name}:{str(error)[:500]}")

    def dump(self) -> dict[str, Any]:
        known=[x for x in self.provider_requests if x["usage_known"]]
        truncations=sum(x.get("finish_reason") in {"length", "max_tokens"} for x in self.provider_requests)
        return {"system":self.output.parent.name,"assigned_ids":self.ids,"completed_ids":[],"model_calls":self.calls,"provider_requests":self.provider_requests,"provider_request_count":len(self.provider_requests),"malformed_tool_retry_requests":sum(x["request_kind"] == "malformed_tool_retry" for x in self.provider_requests),"proxy_observable":True,"cost_complete":all(x["usage_known"] for x in self.provider_requests),"input_tokens":sum(x["input_tokens"] for x in known),"uncached_input_tokens":sum(max(0,x["input_tokens"]-x["cached_input_tokens"]) for x in known),"cached_input_tokens":sum(x["cached_input_tokens"] for x in known),"output_tokens":sum(x["output_tokens"] for x in known),"output_truncation_count":truncations,"output_truncation_hit":truncations>0,"worker_delegations":self.worker_delegations,"terminal_success":self.terminal_success,"terminal_reason":self.terminal_reason,"adk_input_tokens":sum(x["input_tokens"] for x in self.calls),"adk_cached_input_tokens":sum(x["cached_input_tokens"] for x in self.calls),"adk_output_tokens":sum(x["output_tokens"] for x in self.calls),"mcp_calls":self.tools,"worker_calls":self.worker_calls,"tool_calls":len(self.tools),"runtime_seconds":time.monotonic()-self.started,"failures":self.failures,"private_ground_truth_accesses":0}

def write_system_config(run_dir: Path, system: str, runtime_manifest: dict[str, Any], telemetry: dict[str, Any], transcripts: list[dict[str, Any]] | None = None) -> None:
    import csv
    runtime_manifest_path=run_dir/"runtime_manifest.json"; runtime_manifest_path.write_text(json.dumps(runtime_manifest,ensure_ascii=False,indent=2)+"\n")
    (run_dir/"telemetry.json").write_text(json.dumps(telemetry,ensure_ascii=False,indent=2)+"\n")
    if system!="tfidf":
        with (run_dir/"transcript.jsonl").open("x",encoding="utf-8") as f:
            for item in transcripts or []: f.write(json.dumps(item,ensure_ascii=False)+"\n")

def model_costs(telemetry: dict[str, Any], pricing: dict[str, Any]) -> float | None:
    """Cost only observed provider requests; ADK callbacks are telemetry, not billing."""
    requests_present="provider_requests" in telemetry
    requests=telemetry.get("provider_requests",[])
    if not requests_present and not telemetry.get("model_calls") and not telemetry.get("model_calls_count"):
        return 0.0
    if not telemetry.get("proxy_observable", "provider_requests" in telemetry):
        return None
    if any(not request.get("usage_known") for request in requests): return None
    if not requests and telemetry.get("model_calls_count",len(telemetry.get("model_calls",[]))): return None
    return sum(call_cost(request,pricing) for request in requests)
