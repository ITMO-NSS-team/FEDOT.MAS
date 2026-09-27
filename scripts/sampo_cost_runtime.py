"""Shared GT-blind execution, tracing, schema introspection, and preflight."""
from __future__ import annotations
import asyncio, csv, json, os, re, shutil, subprocess, sys, time
import math
from pathlib import Path
from typing import Any
from sampo_cost_demo import OUT, ROOT, read_csv, call_cost

BUDGETS = {"batch_size": 20, "max_model_calls": 30, "max_mcp_calls": 60, "max_prompt_tokens_per_call": 64000, "max_batch_seconds": 300, "max_output_tokens_per_call": 8192}
NEUTRAL_TASK = "Map each assigned historical construction work name to three distinct allowed labels using only the supplied public data and tools. Produce valid top-3 predictions for all assigned IDs."
NEUTRAL_SYSTEM = "You are a single agent completing a batch of SAMPO construction work name mappings. Follow the task and use only public batch data, allowed labels, and available tools. Return valid top-three predictions for every assigned ID."

def required_runtime() -> tuple[str, str, str, str]:
    cheap, strong = os.getenv("SAMPO_CHEAP_MODEL"), os.getenv("SAMPO_CODEX_MODEL")
    endpoint = os.getenv("SAMPO_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    provider = os.getenv("SAMPO_PROVIDER", "openai-compatible" if endpoint else "")
    missing = [name for name, value in (("SAMPO_CHEAP_MODEL", cheap), ("SAMPO_CODEX_MODEL", strong), ("SAMPO_PROVIDER/endpoint", provider and endpoint)) if not value]
    if missing: raise RuntimeError("Missing live inference configuration: " + ", ".join(missing))
    return cheap, strong, provider, endpoint

def pricing_preflight(models: set[str]) -> dict[str, Any]:
    path = OUT / "pricing.json"
    pricing = json.loads(path.read_text(encoding="utf-8"))
    entries = pricing.get("models", {})
    missing = models - entries.keys()
    if missing: raise RuntimeError("Pricing entries missing for: " + ", ".join(sorted(missing)) + "; provider cost is not exposed by the configured ADK adapter, so explicit pricing is required")
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
    if hasattr(value, "model_dump"): return value.model_dump(mode="json", exclude_none=True)
    if hasattr(value, "model_dump_json"): return json.loads(value.model_dump_json(exclude_none=True))
    return str(value)

class RuntimeTrace:
    """ADK plugin compatible callback object with hard pre-call ceilings."""
    def __init__(self, ids: list[str], run_id: str, output: Path, model: str, tool_names: set[str] | None = None, batch: int = 0):
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
        self.tool_names=tool_names or set(); self.calls: list[dict[str, Any]]=[]; self.tools: list[dict[str, Any]]=[]; self.worker_calls: list[dict[str, Any]]=[]; self.provider_requests: list[dict[str, Any]]=[]; self.failures=[]; self.transcript=[]; self.started=time.monotonic(); self.call_started=0.0

    def record_provider_request(self, *, model: str, request_kind: str) -> None:
        self.provider_requests.append({"model": model, "request_kind": request_kind})
        self.transcript.append({"type": "provider_request", "model": model, "request_kind": request_kind})

    async def before_model_callback(self, *, callback_context, llm_request):
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
        if is_mcp and len(self.tools)>=BUDGETS["max_mcp_calls"]: self.failures.append("mcp_call_limit"); raise RuntimeError("mcp_call_limit")
        args=tool_args or {}; ids=args.get("example_ids",[])
        if ids and not set(ids)<=set(self.ids): self.failures.append("out_of_scope_tool_ids"); raise RuntimeError("out_of_scope_tool_ids")
        row={"name":name,"arguments":_jsonable(args)}
        (self.tools if is_mcp else self.worker_calls).append(row)
        self.transcript.append({"type":"tool_request","kind":"mcp" if is_mcp else "worker","name":name,"arguments":_jsonable(args)})

    async def after_tool_callback(self, *, tool, tool_args, tool_context, result):
        self.transcript.append({"type":"tool_response","name":tool.name,"response":_jsonable(result)})

    async def on_tool_error_callback(self, *, tool, tool_args, tool_context, error):
        self.failures.append(f"tool_error:{tool.name}:{str(error)[:500]}")

    def dump(self) -> dict[str, Any]:
        return {"system":self.output.name,"assigned_ids":self.ids,"completed_ids":[],"model_calls":self.calls,"provider_requests":self.provider_requests,"provider_request_count":len(self.provider_requests),"malformed_tool_retry_requests":sum(x["request_kind"] == "malformed_tool_retry" for x in self.provider_requests),"mcp_calls":self.tools,"worker_calls":self.worker_calls,"tool_calls":len(self.tools),"runtime_seconds":time.monotonic()-self.started,"failures":self.failures,"private_ground_truth_accesses":0}

def write_system_config(run_dir: Path, system: str, runtime_manifest: dict[str, Any], telemetry: dict[str, Any], transcripts: list[dict[str, Any]] | None = None) -> None:
    import csv
    runtime_manifest_path=run_dir/"runtime_manifest.json"; runtime_manifest_path.write_text(json.dumps(runtime_manifest,ensure_ascii=False,indent=2)+"\n")
    (run_dir/"telemetry.json").write_text(json.dumps(telemetry,ensure_ascii=False,indent=2)+"\n")
    if system!="tfidf":
        with (run_dir/"transcript.jsonl").open("x",encoding="utf-8") as f:
            for item in transcripts or []: f.write(json.dumps(item,ensure_ascii=False)+"\n")

def model_costs(telemetry: dict[str, Any], pricing: dict[str, Any]) -> float:
    return sum(call_cost(call, pricing) for call in telemetry.get("model_calls", []))
