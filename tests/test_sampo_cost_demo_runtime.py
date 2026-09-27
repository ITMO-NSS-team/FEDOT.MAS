from __future__ import annotations
import csv, json, os, subprocess, sys
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import sampo_cost_demo as demo
from sampo_cost_runtime import NEUTRAL_SYSTEM, NEUTRAL_TASK, required_runtime, pricing_preflight
from run_sampo_cost_demo import neutral_user

def test_operational_set_disjoint_from_pilot_and_final_by_id_and_name():
    operational=demo.read_csv(demo.OUT/"operational_inputs.csv")
    final=demo.read_csv(demo.OUT/"public_inputs.csv")
    pilot=demo.read_csv(ROOT/"artifacts/sampo_benchmark/pilot_inputs.csv")
    assert len(operational)==20
    assert not ({r["example_id"] for r in operational}&{r["example_id"] for r in final})
    assert not ({demo.normalized_name(r["raw_work_name"]) for r in operational}&{demo.normalized_name(r["raw_work_name"]) for r in final})
    assert not ({r["example_id"] for r in operational}&{r["example_id"] for r in pilot})
    assert not ({demo.normalized_name(r["raw_work_name"]) for r in operational}&{demo.normalized_name(r["raw_work_name"]) for r in pilot})
    manifest=json.loads((demo.OUT/"manifest.json").read_text())
    assert manifest["operational_set"]["sha256"]==demo.sha256(demo.OUT/"operational_inputs.csv")

def test_neutral_prompts_are_identical_and_gt_blind():
    assert NEUTRAL_SYSTEM == "You are a single agent completing a batch of SAMPO construction work name mappings. Follow the task and use only public batch data, allowed labels, and available tools. Return valid top-three predictions for every assigned ID."
    assert NEUTRAL_TASK == "Map each assigned historical construction work name to three distinct allowed labels using only the supplied public data and tools. Produce valid top-3 predictions for all assigned IDs."
    rows=[{"example_id":"x","raw_work_name":"Публичная работа"}];labels=["Метка"]
    prompt_b=neutral_user(rows,labels); prompt_d=neutral_user(rows,labels)
    assert prompt_b.encode()==prompt_d.encode()
    assert "private_ground_truth" not in prompt_b and "ground_truth" not in prompt_b

def test_phase5_historical_sources_unchanged():
    paths=["scripts/sampo_phase_5_policy.py","scripts/run_sampo_phase_5_full.py","artifacts/sampo_phase_5/qualification_terminal_result.json"]
    for path in paths:
        subprocess.run(["git","diff","--quiet","HEAD","--",path],cwd=ROOT,check=True)

def test_neutral_server_tool_schemas_are_exactly_six():
    server=ROOT/"mcp-servers/sampo-cost-demo/src/mcp_sampo_cost_demo/server.py"
    source=server.read_text()
    for name in ("list_methods","prepare_candidates","inspect_candidates","save_default_top3","save_ranked_top3","get_prediction_status"):
        assert f"def {name}(" in source
    assert "pilot_inputs.csv" not in source

@pytest.mark.asyncio
async def test_b_and_d_introspect_identical_real_neutral_schemas(tmp_path):
    from sampo_cost_runtime import scoped_server, introspect
    rows=demo.read_csv(demo.OUT/"operational_inputs.csv")[:20]
    ids=[r["example_id"] for r in rows]
    registry_a=scoped_server(ids,"schema-a",tmp_path,demo.OUT/"operational_inputs.csv")
    registry_b=scoped_server(ids,"schema-b",tmp_path,demo.OUT/"operational_inputs.csv")
    _,schema_a=await introspect(registry_a,"sampo-cost-demo")
    _,schema_b=await introspect(registry_b,"sampo-cost-demo")
    assert schema_a==schema_b
    assert {tool["name"] for tool in schema_a}=={"list_methods","prepare_candidates","inspect_candidates","save_default_top3","save_ranked_top3","get_prediction_status"}

@pytest.mark.asyncio
async def test_new_server_handles_arbitrary_scoped_ids_and_write_once(tmp_path):
    from mcp import StdioServerParameters
    from mcp.client.session import ClientSession
    from mcp.client.stdio import stdio_client
    from sampo_cost_runtime import scoped_server
    rows=demo.read_csv(demo.OUT/"operational_inputs.csv")[:2];ids=[r["example_id"] for r in rows]
    reg=scoped_server(ids,"arbitrary-scope",tmp_path,demo.OUT/"operational_inputs.csv")
    cfg=reg["sampo-cost-demo"]
    async with stdio_client(StdioServerParameters(command=cfg.command,args=list(cfg.args),env=cfg.env)) as (rs,ws):
        async with ClientSession(rs,ws) as client:
            await client.initialize()
            methods=await client.call_tool("list_methods",{})
            assert not methods.isError
            prep=await client.call_tool("prepare_candidates",{"methods":["char_tfidf"],"k":5,"fusion":"rrf"})
            assert not prep.isError
            data=prep.structuredContent
            aid=data["artifact_id"]
            saved=await client.call_tool("save_default_top3",{"run_id":"arbitrary-scope","artifact_id":aid,"example_ids":ids})
            assert not saved.isError
            repeated=await client.call_tool("save_default_top3",{"run_id":"arbitrary-scope","artifact_id":aid,"example_ids":ids})
            assert not repeated.isError
            assert repeated.structuredContent["already_identical"]==2
            status=await client.call_tool("get_prediction_status",{})
            assert status.structuredContent["stored_ids"]==ids

def test_preflight_rejects_null_model_ids(monkeypatch):
    monkeypatch.delenv("SAMPO_CHEAP_MODEL",raising=False);monkeypatch.delenv("SAMPO_CODEX_MODEL",raising=False)
    with pytest.raises(RuntimeError,match="SAMPO_CHEAP_MODEL"):
        required_runtime()

def test_pricing_preflight_rejects_empty_config(monkeypatch,tmp_path):
    monkeypatch.setattr("sampo_cost_runtime.OUT",tmp_path)
    (tmp_path/"pricing.json").write_text('{"models":{}}')
    with pytest.raises(RuntimeError,match="Pricing entries missing"):
        pricing_preflight({"m"})

def test_pricing_preflight_accepts_valid_explicit_prices(monkeypatch,tmp_path):
    monkeypatch.setattr("sampo_cost_runtime.OUT",tmp_path)
    (tmp_path/"pricing.json").write_text(json.dumps({"models":{"cheap":{"input_usd_per_1m":1.0,"output_usd_per_1m":2.0,"source":"provider pricing","source_date":"2026-09-01"},"codex":{"input_usd_per_1m":3.0,"output_usd_per_1m":4.0,"source":"provider pricing","source_date":"2026-09-01"}}}))
    assert set(pricing_preflight({"cheap","codex"})["models"])=={"cheap","codex"}

def test_live_telemetry_counts_provider_retries_separately_and_model_usage(monkeypatch,tmp_path):
    from types import SimpleNamespace
    from sampo_cost_runtime import RuntimeTrace
    trace=RuntimeTrace(["x"],"run",tmp_path/"cheap_single_agent"/"batch_0000","m")
    trace.record_provider_request(model="m",request_kind="initial")
    trace.record_provider_request(model="m",request_kind="malformed_tool_retry")
    async def capture(agent, prompt, completion):
        await trace.after_model_callback(callback_context=SimpleNamespace(agent_name=agent),llm_response=SimpleNamespace(usage_metadata=SimpleNamespace(prompt_token_count=prompt,candidates_token_count=completion,cached_content_token_count=2)))
    import asyncio
    asyncio.run(capture("coordinator",10,4));asyncio.run(capture("worker",20,5))
    data=trace.dump()
    assert data["provider_request_count"]==2
    assert data["malformed_tool_retry_requests"]==1
    assert sum(x["input_tokens"] for x in data["model_calls"])>0
    assert {x["agent_name"] for x in data["model_calls"]}=={"coordinator","worker"}

def test_all_system_prediction_schema_is_common():
    source=(ROOT/"scripts/run_sampo_cost_demo.py").read_text()
    for system in ("tfidf","cheap_single_agent","fedotmas_cost_aware","codex"):
        assert system in source
    assert 'fieldnames=["example_id","top_1","top_2","top_3"]' in source

def test_runtime_server_environment_never_includes_gt_path():
    from sampo_cost_runtime import scoped_server
    rows=demo.read_csv(demo.OUT/"operational_inputs.csv")[:2]
    config=scoped_server([r["example_id"] for r in rows],"envcheck",demo.OUT/".envcheck",demo.OUT/"operational_inputs.csv")
    env=next(iter(config.values())).env
    assert not any("GROUND_TRUTH" in key.upper() or "PRIVATE_GT" in key.upper() for key in env)
    assert all("private_ground_truth" not in value for value in env.values())
