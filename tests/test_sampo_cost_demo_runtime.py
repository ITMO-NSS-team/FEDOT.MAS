from __future__ import annotations
import csv, json, os, subprocess, sys
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import sampo_cost_demo as demo
from sampo_cost_runtime import NEUTRAL_SYSTEM, NEUTRAL_TASK, required_runtime, pricing_preflight
from run_sampo_cost_demo import neutral_user, PHASE5_CONFIG, PHASE5_TASK, smoke_system_failed

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

def test_neutral_prompts_are_identical_without_inline_labels_and_gt_blind():
    assert NEUTRAL_SYSTEM == "You are a single agent completing a batch of SAMPO construction work name mappings. Follow the task and use only public batch data, allowed labels, and available tools. Return valid top-three predictions for every assigned ID."
    assert NEUTRAL_TASK == "Map each assigned historical construction work name to three distinct allowed labels using only the supplied public data and tools. Produce valid top-3 predictions for all assigned IDs."
    rows=[{"example_id":"x","raw_work_name":"Публичная работа"}]
    prompt_b=neutral_user(rows,"same-run"); prompt_d=neutral_user(rows,"same-run")
    assert prompt_b.encode()==prompt_d.encode()
    assert "private_ground_truth" not in prompt_b and "ground_truth" not in prompt_b
    assert "Метка" not in prompt_b and "Allowed target labels:" not in prompt_b
    assert "Persistence run ID: same-run" in prompt_b
    assert "at most 10 example IDs" in prompt_b

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
    by_name={tool["name"]:tool for tool in schema_a}
    inspect_ids=by_name["inspect_candidates"]["inputSchema"]["properties"]["example_ids"]
    assert inspect_ids["minItems"]==1 and inspect_ids["maxItems"]==10
    assert "larger than 10 IDs must be split" in inspect_ids["description"]
    required_run_id="Exact persistence run ID supplied by the harness in the task message; do not invent or substitute another identifier."
    for name in ("save_default_top3","save_ranked_top3"):
        assert required_run_id in by_name[name]["inputSchema"]["properties"]["run_id"]["description"]

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

@pytest.mark.asyncio
async def test_neutral_prepare_and_inspect_payloads_are_bounded_for_worst_case(tmp_path):
    from mcp import StdioServerParameters
    from mcp.client.session import ClientSession
    from mcp.client.stdio import stdio_client
    from sampo_cost_runtime import scoped_server
    rows=demo.read_csv(demo.OUT/"operational_inputs.csv"); ids=[r["example_id"] for r in rows]
    reg=scoped_server(ids,"bounded-neutral",tmp_path,demo.OUT/"operational_inputs.csv")
    cfg=reg["sampo-cost-demo"]
    async with stdio_client(StdioServerParameters(command=cfg.command,args=list(cfg.args),env=cfg.env)) as (rs,ws):
        async with ClientSession(rs,ws) as client:
            await client.initialize()
            prepared=await client.call_tool("prepare_candidates",{"methods":["char_tfidf","construction_token_tfidf","word_tfidf","bm25_token","char_word_fusion"],"k":50,"fusion":"rrf"})
            assert not prepared.isError
            result=prepared.structuredContent
            assert len(json.dumps(result,ensure_ascii=False).encode("utf-8"))<15_000
            assert len(result["examples"])==20
            assert set(result["examples"][0])=={"example_id","fused_top_3","method_count","distinct_top1_labels","top_1_vote_count","top1_top2_margin"}
            artifact_path=Path(cfg.env["SAMPO_RUN_DIR"])/".candidate_artifacts"/"bounded-neutral"/(result["artifact_id"]+".json")
            artifact=json.loads(artifact_path.read_text(encoding="utf-8"))
            assert len(artifact["examples"][0]["fused_candidates"])>3
            assert len(artifact["examples"][0]["method_candidates"])>3
            inspected=await client.call_tool("inspect_candidates",{"artifact_id":result["artifact_id"],"example_ids":[ids[0]],"candidate_limit":7})
            assert not inspected.isError
            payload=inspected.structuredContent
            assert len(json.dumps(payload,ensure_ascii=False).encode("utf-8"))<5_000
            assert set(payload["examples"][0])=={"example_id","fused_candidates"}
            assert len(payload["examples"][0]["fused_candidates"])==7
            assert all(set(candidate)=={"candidate_index","label","fused_rank","fusion_score","support_count","best_rank","methods"} for candidate in payload["examples"][0]["fused_candidates"])

@pytest.mark.asyncio
async def test_phase5_adapter_contract_is_gt_blind_bounded_and_durable(tmp_path):
    from mcp import StdioServerParameters
    from mcp.client.session import ClientSession
    from mcp.client.stdio import stdio_client
    from sampo_cost_runtime import scoped_server
    rows=demo.read_csv(demo.OUT/"operational_inputs.csv")[:2]; ids=[r["example_id"] for r in rows]
    reg=scoped_server(ids,"phase5-contract",tmp_path,demo.OUT/"operational_inputs.csv",phase5=True)
    cfg=reg["sampo-cost-demo-phase5"]
    assert all("private_ground_truth" not in value for value in cfg.env.values())
    async with stdio_client(StdioServerParameters(command=cfg.command,args=list(cfg.args),env=cfg.env)) as (rs,ws):
        async with ClientSession(rs,ws) as client:
            await client.initialize()
            listed=await client.call_tool("list_methods",{})
            assert not listed.isError
            prepared=await client.call_tool("prepare_candidate_batch",{"offset":0,"limit":2,"methods":["char_tfidf","construction_token_tfidf","word_tfidf","bm25_token","char_word_fusion"],"k":5,"fusion":"rrf"})
            assert not prepared.isError
            summary=prepared.structuredContent
            assert len(json.dumps(summary,ensure_ascii=False).encode("utf-8"))<5_000
            partition=await client.call_tool("partition_candidate_batch",{"artifact_id":summary["artifact_id"],"example_ids":ids})
            assert not partition.isError
            groups=partition.structuredContent
            outside=await client.call_tool("partition_candidate_batch",{"artifact_id":summary["artifact_id"],"example_ids":[*ids,"outside-scope"]})
            assert outside.isError
            assert set(groups["review_ids"])|set(groups["fallback_ids"])==set(ids)
            assert not set(groups["review_ids"])&set(groups["fallback_ids"])
            assert groups["review_ids"] and groups["fallback_ids"]
            evidence=await client.call_tool("get_candidate_evidence",{"artifact_id":summary["artifact_id"],"example_ids":groups["review_ids"],"candidate_limit":10,"selection":"fused"})
            assert not evidence.isError
            evidence_data=evidence.structuredContent
            assert len(json.dumps(evidence_data,ensure_ascii=False).encode("utf-8"))<10_000
            decisions=[]
            for example in evidence_data["examples"]:
                decisions.append({"example_id":example["example_id"],"candidate_indices":[item["candidate_index"] for item in example["fused_candidates"][:3]]})
            saved_review=await client.call_tool("save_review_decisions",{"run_id":"phase5-contract","artifact_id":summary["artifact_id"],"decisions":decisions})
            saved_fallback=await client.call_tool("save_candidate_predictions",{"run_id":"phase5-contract","artifact_id":summary["artifact_id"],"example_ids":groups["fallback_ids"]})
            assert not saved_review.isError and not saved_fallback.isError
            repeated_review=await client.call_tool("save_review_decisions",{"run_id":"phase5-contract","artifact_id":summary["artifact_id"],"decisions":decisions})
            repeated_fallback=await client.call_tool("save_candidate_predictions",{"run_id":"phase5-contract","artifact_id":summary["artifact_id"],"example_ids":groups["fallback_ids"]})
            assert repeated_review.structuredContent["already_identical"]==len(decisions)
            assert repeated_fallback.structuredContent["already_identical"]==len(groups["fallback_ids"])
            status=await client.call_tool("get_prediction_status",{})
            assert not status.isError
            assert status.structuredContent["stored_ids"]==ids
            assert status.structuredContent["missing_ids"]==[]

def test_preflight_rejects_null_model_ids(monkeypatch):
    monkeypatch.delenv("SAMPO_FEDOT_MODEL",raising=False);monkeypatch.delenv("SAMPO_TERRA_MODEL",raising=False)
    with pytest.raises(RuntimeError,match="SAMPO_FEDOT_MODEL"):
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
    trace=RuntimeTrace(["x"],"run",tmp_path/"terra_single_agent"/"batch_0000","m")
    trace.record_provider_request({"request_id":"one","model":"m","request_kind":"initial","success":True,"input_tokens":1,"cached_input_tokens":0,"output_tokens":1,"finish_reason":"stop","elapsed_seconds":0.1,"usage_known":True})
    trace.record_provider_request({"request_id":"two","model":"m","request_kind":"malformed_tool_retry","success":True,"input_tokens":1,"cached_input_tokens":0,"output_tokens":1,"finish_reason":"stop","elapsed_seconds":0.1,"usage_known":True})
    async def capture(agent, prompt, completion):
        await trace.after_model_callback(callback_context=SimpleNamespace(agent_name=agent),llm_response=SimpleNamespace(usage_metadata=SimpleNamespace(prompt_token_count=prompt,candidates_token_count=completion,cached_content_token_count=2)))
    import asyncio
    asyncio.run(capture("coordinator",10,4));asyncio.run(capture("worker",20,5))
    data=trace.dump()
    assert data["provider_request_count"]==2
    assert data["malformed_tool_retry_requests"]==1
    assert sum(x["input_tokens"] for x in data["model_calls"])>0
    assert {x["agent_name"] for x in data["model_calls"]}=={"coordinator","worker"}

def test_jsonable_keeps_nested_mapping_arguments_structured():
    from sampo_cost_runtime import _jsonable
    from collections import UserDict
    assert _jsonable(UserDict({"artifact_id":"a","example_ids":["1","2"]}))=={"artifact_id":"a","example_ids":["1","2"]}

@pytest.mark.asyncio
async def test_phase5_blocks_second_worker_delegation_and_records_truncation(tmp_path):
    from types import SimpleNamespace
    from sampo_cost_runtime import RuntimeTrace
    trace=RuntimeTrace(["1","2"],"run",tmp_path/"fedotmas_cost_aware"/"batch_0000","m")
    tool=SimpleNamespace(name="construction_batch_specialist")
    args={"artifact_id":"a","example_ids":["1","2"]}
    await trace.before_tool_callback(tool=tool,tool_args=args,tool_context=None)
    assert trace.worker_calls[0]["arguments"]==args
    with pytest.raises(RuntimeError,match="duplicate_worker_delegation"):
        await trace.before_tool_callback(tool=tool,tool_args=args,tool_context=None)
    trace.record_provider_request({"request_id":"r","model":"m","request_kind":"initial","success":True,"input_tokens":3,"cached_input_tokens":0,"output_tokens":8192,"finish_reason":"length","elapsed_seconds":1,"usage_known":True})
    data=trace.dump()
    assert data["worker_delegations"]==2
    assert "duplicate_worker_delegation" in data["failures"]
    assert data["output_truncation_count"]==1 and data["output_truncation_hit"] is True

@pytest.mark.asyncio
async def test_complementary_phase5_persistence_terminates_invocation_successfully(tmp_path):
    from types import SimpleNamespace
    from sampo_cost_runtime import RuntimeTrace
    ids=[str(i) for i in range(20)]; labels={"A","B","C"}
    output=tmp_path/"fedotmas_cost_aware"/"batch_0000";output.mkdir(parents=True)
    trace=RuntimeTrace(ids,"run",output,"m",{"save_review_decisions","save_candidate_predictions"},allowed_labels=labels)
    invocation=SimpleNamespace(end_invocation=False)
    context=SimpleNamespace(_invocation_context=invocation,actions=SimpleNamespace(end_of_agent=False))
    tool=SimpleNamespace(name="save_review_decisions")
    def write_rows(selected):
        with (output/".predictions.jsonl").open("a",encoding="utf-8") as stream:
            for eid in selected:
                stream.write(json.dumps({"example_id":eid,"top_1":"A","top_2":"B","top_3":"C"})+"\n")
    write_rows(ids[:10])
    await trace.after_tool_callback(tool=tool,tool_args={},tool_context=context,result={"isError":False})
    assert not invocation.end_invocation and not trace.terminal_success
    write_rows(ids[10:])
    await trace.after_tool_callback(tool=SimpleNamespace(name="save_candidate_predictions"),tool_args={},tool_context=context,result={"isError":False})
    assert trace.terminal_success is True
    assert trace.terminal_reason=="durable_batch_complete"
    assert invocation.end_invocation is True and context.actions.end_of_agent is True
    assert trace.dump()["failures"]==[]

@pytest.mark.asyncio
async def test_completed_batch_prevents_later_model_turn_or_worker_delegation(tmp_path):
    from types import SimpleNamespace
    from sampo_cost_runtime import RuntimeTrace
    ids=[str(i) for i in range(20)]; output=tmp_path/"fedotmas_cost_aware"/"batch_0000";output.mkdir(parents=True)
    trace=RuntimeTrace(ids,"run",output,"m",allowed_labels={"A","B","C"})
    with (output/".predictions.jsonl").open("w",encoding="utf-8") as stream:
        for eid in ids: stream.write(json.dumps({"example_id":eid,"top_1":"A","top_2":"B","top_3":"C"})+"\n")
    worker_invocation=SimpleNamespace(end_invocation=False)
    worker_context=SimpleNamespace(_invocation_context=worker_invocation,actions=SimpleNamespace(end_of_agent=False))
    await trace.after_tool_callback(tool=SimpleNamespace(name="save_candidate_predictions"),tool_args={},tool_context=worker_context,result={"isError":False})
    assert worker_invocation.end_invocation is True
    parent_invocation=SimpleNamespace(end_invocation=False)
    parent_context=SimpleNamespace(_invocation_context=parent_invocation,actions=SimpleNamespace(end_of_agent=False))
    await trace.after_tool_callback(tool=SimpleNamespace(name="construction_batch_specialist"),tool_args={},tool_context=parent_context,result={"isError":False})
    assert parent_invocation.end_invocation is True and parent_context.actions.end_of_agent is True
    await trace.before_model_callback(callback_context=SimpleNamespace(_invocation_context=parent_invocation),llm_request=SimpleNamespace(model="m"))
    assert parent_invocation.end_invocation is True
    assert trace.worker_delegations==0 and trace.calls==[]

def test_durable_terminal_success_is_not_classified_as_smoke_failure():
    info={"completed_examples":20,"cost_complete":True,"output_truncation_hit":False,"safety_ceiling_hit":False,"failures":[{"error":"duplicate_worker_delegation"},{"error":"execution_error after durable_batch_complete: duplicate_worker_delegation"}],"terminal_success":True}
    assert smoke_system_failed(info,phase5=True) is False
    assert smoke_system_failed({**info,"failures":["model_call_limit"]},phase5=True) is True

def test_all_system_prediction_schema_is_common():
    source=(ROOT/"scripts/run_sampo_cost_demo.py").read_text()
    for system in ("tfidf","fedotmas_cost_aware","terra_single_agent"):
        assert system in source
    assert "cheap_single_agent" not in source
    assert "codex" not in source.lower()
    assert 'fieldnames=["example_id","top_1","top_2","top_3"]' in source

def test_terra_is_a_standalone_openai_tool_loop_with_isolation_metadata():
    import ast
    path=ROOT/"scripts/sampo_terra_standalone.py"
    source=path.read_text()
    tree=ast.parse(source)
    imported=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Import): imported.extend(alias.name for alias in node.names)
        elif isinstance(node,ast.ImportFrom): imported.append(node.module or "")
    assert not any(name.startswith("fedotmas") for name in imported)
    assert "AsyncOpenAI" in source and "ClientSession" in source and "stdio_client" in source
    assert '"harness": "standalone_openai_tool_loop"' in source
    assert '"fedotmas_dependency": False' in source
    assert '"phase5_config_exposed": False' in source
    assert '"fresh_mcp_process": True' in source and '"fresh_model_conversation": True' in source
    assert '"system": "terra_single_agent"' in source

def test_terra_runner_branches_before_fedotmas_helpers():
    source=(ROOT/"scripts/run_sampo_cost_demo.py").read_text()
    body=source[source.index("async def run_agent"):source.index("async def get_manifest")]
    assert body.index('if system == "terra_single_agent"') < body.index("from fedotmas")
    assert "create_toolset" not in body[body.index('if system == "terra_single_agent"'):body.index("from google.adk.agents")]

def test_runtime_server_environment_never_includes_gt_path():
    from sampo_cost_runtime import scoped_server
    rows=demo.read_csv(demo.OUT/"operational_inputs.csv")[:2]
    config=scoped_server([r["example_id"] for r in rows],"envcheck",demo.OUT/".envcheck",demo.OUT/"operational_inputs.csv")
    env=next(iter(config.values())).env
    assert not any("GROUND_TRUTH" in key.upper() or "PRIVATE_GT" in key.upper() for key in env)
    assert all("private_ground_truth" not in value for value in env.values())

def test_runner_uses_tracked_frozen_phase5_inputs():
    runner=(ROOT/"scripts/run_sampo_cost_demo.py").read_text()
    frozen=demo.OUT/"frozen_phase5"
    provenance=json.loads((frozen/"provenance.json").read_text())
    assert PHASE5_CONFIG==(frozen/"config.json")
    assert PHASE5_TASK==(frozen/"task.txt").read_text()
    assert "artifacts/sampo_phase_5/structural_review" not in runner
    assert provenance["behavioral_pass"] is True
    assert provenance["selection_used_new_cost_demo_ground_truth"] is False
    assert provenance["config_sha256"]==demo.sha256(frozen/"config.json")
    assert provenance["task_sha256"]==demo.sha256(frozen/"task.txt")

def test_model_config_uses_explicit_preflight_endpoint_and_proxy(monkeypatch):
    from sampo_cost_runtime import model_config_for
    from fedotmas.common.llm import _ProxyClient, make_llm
    monkeypatch.setenv("SAMPO_API_KEY","test-key")
    config=model_config_for("cheap-model","http://proxy.example/v1")
    llm=make_llm(config)
    assert config.api_base=="http://proxy.example/v1"
    assert isinstance(llm.llm_client,_ProxyClient)
    assert str(llm.llm_client._client.base_url).startswith("http://proxy.example/v1")

class _FakeProviderResponse:
    def __init__(self,prompt,completion,cached=0,malformed=False):
        from types import SimpleNamespace
        self.usage=SimpleNamespace(prompt_tokens=prompt,completion_tokens=completion,prompt_tokens_details=SimpleNamespace(cached_tokens=cached))
        call={"id":"call-1","type":"function","function":{"name":"tool","arguments":"{bad" if malformed else "{}"}}
        message={"role":"assistant","content":None,"tool_calls":[call] if malformed else []}
        self.choices=[SimpleNamespace(finish_reason="tool_calls" if malformed else "stop")]
        self._payload={"id":"resp","object":"chat.completion","created":1,"model":"m","choices":[{"index":0,"finish_reason":"tool_calls" if malformed else "stop","message":message}],"usage":{"prompt_tokens":prompt,"completion_tokens":completion,"total_tokens":prompt+completion,"prompt_tokens_details":{"cached_tokens":cached}}}
    def model_dump(self): return self._payload

class _FakeCompletions:
    def __init__(self,responses): self.responses=list(responses)
    async def create(self,**kwargs):
        response=self.responses.pop(0)
        if isinstance(response,Exception): raise response
        return response

def _fake_proxy(responses):
    from types import SimpleNamespace
    from fedotmas.common.llm import _ProxyClient
    proxy=object.__new__(_ProxyClient)
    proxy._extra_body={}
    proxy._client=SimpleNamespace(chat=SimpleNamespace(completions=_FakeCompletions(responses)))
    return proxy

@pytest.mark.asyncio
async def test_proxy_records_normal_provider_usage_lifecycle():
    from fedotmas.common.llm import _ProxyClient
    events=[]; previous=_ProxyClient.request_observer; _ProxyClient.request_observer=events.append
    try:
        result=await _fake_proxy([_FakeProviderResponse(100,20,5)]).acompletion("m",[],[],temperature=0)
    finally: _ProxyClient.request_observer=previous
    assert result is not None and len(events)==1
    event=events[0]
    assert event["request_kind"]=="initial" and event["success"] is True
    assert event["usage_known"] is True and event["input_tokens"]==100
    assert event["cached_input_tokens"]==5 and event["output_tokens"]==20
    assert event["finish_reason"]=="stop" and event["elapsed_seconds"]>=0
    assert event["request_id"]

@pytest.mark.asyncio
async def test_proxy_retry_usage_is_costed_and_not_double_counted_with_adk():
    from types import SimpleNamespace
    from fedotmas.common.llm import _ProxyClient
    from sampo_cost_runtime import RuntimeTrace, model_costs
    events=[]; previous=_ProxyClient.request_observer; _ProxyClient.request_observer=events.append
    try:
        await _fake_proxy([_FakeProviderResponse(1000,100,100,malformed=True),_FakeProviderResponse(1200,80)]).acompletion("m",[],[{"type":"function"}])
    finally: _ProxyClient.request_observer=previous
    assert [x["request_kind"] for x in events]==["initial","malformed_tool_retry"]
    assert [x["input_tokens"] for x in events]==[1000,1200]
    trace=RuntimeTrace(["x"],"run",Path("batch"),"m")
    async def adk_callback():
        await trace.after_model_callback(callback_context=SimpleNamespace(agent_name="worker"),llm_response=SimpleNamespace(usage_metadata=SimpleNamespace(prompt_token_count=9999,candidates_token_count=9999,cached_content_token_count=0)))
    import asyncio
    await adk_callback()
    pricing={"models":{"m":{"input_usd_per_1m":1.0,"cached_input_usd_per_1m":0.5,"output_usd_per_1m":2.0}}}
    cost=model_costs({"proxy_observable":True,"provider_requests":events,"model_calls":trace.calls,"model_calls_count":1},pricing)
    assert cost==pytest.approx(0.00251)

@pytest.mark.asyncio
async def test_proxy_failed_request_marks_unknown_usage_and_incomplete_cost():
    from fedotmas.common.llm import _ProxyClient
    from sampo_cost_runtime import RuntimeTrace, model_costs
    events=[]; previous=_ProxyClient.request_observer; _ProxyClient.request_observer=events.append
    try:
        with pytest.raises(RuntimeError,match="provider down"):
            await _fake_proxy([RuntimeError("provider down")]).acompletion("m",[],[])
    finally: _ProxyClient.request_observer=previous
    assert len(events)==1 and events[0]["success"] is False
    assert events[0]["usage_known"] is False and events[0]["input_tokens"] is None
    trace=RuntimeTrace(["x"],"run",Path("batch"),"m")
    trace.record_provider_request(events[0])
    assert trace.dump()["cost_complete"] is False
    assert model_costs({"proxy_observable":True,"provider_requests":events,"model_calls_count":1},{"models":{}}) is None
