"""Standalone OpenAI-compatible tool loop for the neutral SAMPO baseline.

This module deliberately has no FEDOT.MAS imports or runtime dependencies.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

from mcp import StdioServerParameters
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client
from openai import AsyncOpenAI


def _openai_tool(tool: Any) -> dict[str, Any]:
    schema = tool.inputSchema or {"type": "object", "properties": {}}
    return {"type": "function", "function": {
        "name": tool.name,
        "description": tool.description or "",
        "parameters": schema,
    }}


def _content(result: Any) -> str:
    chunks = []
    for item in getattr(result, "content", []) or []:
        if getattr(item, "text", None) is not None:
            chunks.append(item.text)
        elif hasattr(item, "model_dump"):
            chunks.append(json.dumps(item.model_dump(mode="json"), ensure_ascii=False))
    if not chunks and getattr(result, "structuredContent", None) is not None:
        return json.dumps(result.structuredContent, ensure_ascii=False)
    return "\n".join(chunks)


async def run(*, model: str, endpoint: str, api_key: str, rows: list[dict[str, str]],
              run_id: str, output_dir: Path, public_file: Path, labels_file: Path,
              system_prompt: str, user_prompt: str, budgets: dict[str, int],
              git_commit: str) -> dict[str, Any]:
    """Run Terra with a fresh direct MCP session and provider conversation."""
    ids = [row["example_id"] for row in rows]
    env = {k: v for k, v in os.environ.items()
           if k in {"PATH", "TMPDIR", "TEMP", "LANG", "LC_ALL", "VIRTUAL_ENV"}}
    env.update({
        "SAMPO_PUBLIC_INPUTS": str(public_file.resolve()),
        "SAMPO_ALLOWED_LABELS": str(labels_file.resolve()),
        "SAMPO_ASSIGNED_IDS": ",".join(ids), "SAMPO_RUN_ID": run_id,
        "SAMPO_RUN_DIR": str(output_dir.resolve()),
    })
    server_script = Path(__file__).resolve().parents[1] / "mcp-servers/sampo-cost-demo/src/mcp_sampo_cost_demo/server.py"
    server_python = server_script.parents[3] / "sampo-phase6/.venv/bin/python"
    params = StdioServerParameters(command=str(server_python), args=[str(server_script)], env=env)
    client = AsyncOpenAI(api_key=api_key, base_url=endpoint)
    output_dir.mkdir(parents=True, exist_ok=False)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    provider_requests: list[dict[str, Any]] = []
    mcp_calls: list[dict[str, Any]] = []
    transcript: list[dict[str, Any]] = [{"type": "messages_initialized", "messages": messages}]
    failures: list[str] = []
    started = time.monotonic()
    tool_schemas: list[dict[str, Any]] = []
    completed_ids: list[str] = []
    safety_ceiling_hit = False
    try:
        async with asyncio.timeout(budgets["max_batch_seconds"]):
            async with stdio_client(params) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as mcp:
                    await mcp.initialize()
                    listed = await mcp.list_tools()
                    tool_schemas = [_openai_tool(tool) for tool in listed.tools]
                    exposed = {item["function"]["name"] for item in tool_schemas}
                    required = {"list_methods", "prepare_candidates", "inspect_candidates",
                                "save_default_top3", "save_ranked_top3", "get_prediction_status"}
                    if exposed != required:
                        raise RuntimeError(f"Neutral MCP tool surface mismatch: {sorted(exposed)}")
                    transcript.append({"type": "tool_schemas", "tools": tool_schemas})
    
                    async def call_tool(name: str, arguments: dict[str, Any]) -> Any:
                        if len(mcp_calls) >= budgets["max_mcp_calls"]:
                            raise RuntimeError("mcp_call_limit")
                        tool_started = time.monotonic()
                        entry = {"name": name, "arguments": arguments}
                        try:
                            result = await mcp.call_tool(name, arguments)
                            entry["response"] = result.model_dump(mode="json", exclude_none=True)
                            entry["is_error"] = bool(getattr(result, "isError", False))
                            if entry["is_error"]:
                                failures.append("mcp_tool_error:" + name)
                            return result
                        except Exception as exc:
                            entry["is_error"] = True
                            entry["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
                            failures.append("mcp_tool_error:" + name + ":" + entry["error"])
                            raise
                        finally:
                            entry["elapsed_seconds"] = time.monotonic() - tool_started
                            mcp_calls.append(entry)
                            transcript.append({"type": "tool_call", **entry})
    
                    while True:
                        prompt_estimate = len(json.dumps(messages, ensure_ascii=False)) // 4
                        if prompt_estimate > budgets["max_prompt_tokens_per_call"]:
                            safety_ceiling_hit = True
                            failures.append("prompt_token_limit")
                            break
                        if len(provider_requests) >= budgets["max_model_calls"]:
                            safety_ceiling_hit = True
                            failures.append("model_call_limit")
                            break
                        if time.monotonic() - started >= budgets["max_batch_seconds"]:
                            safety_ceiling_hit = True
                            failures.append("runtime_limit")
                            break
                        request_id = f"terra-{len(provider_requests)+1}"
                        call_started = time.monotonic()
                        request = {"request_id": request_id, "model": model,
                                   "request_kind": "initial" if not provider_requests else "tool_followup",
                                   "success": False, "input_tokens": 0, "cached_input_tokens": 0,
                                   "output_tokens": 0, "provider_cost_usd": None,
                                   "finish_reason": None, "usage_known": False}
                        try:
                            response = await client.chat.completions.create(
                                model=model, messages=messages, tools=tool_schemas,
                                tool_choice="auto", max_tokens=budgets["max_output_tokens_per_call"],
                            )
                            usage = response.usage
                            if usage is not None:
                                request["usage_known"] = True
                                request["input_tokens"] = int(usage.prompt_tokens or 0)
                                request["output_tokens"] = int(usage.completion_tokens or 0)
                                details = getattr(usage, "prompt_tokens_details", None)
                                request["cached_input_tokens"] = int(getattr(details, "cached_tokens", 0) or 0)
                            raw = response.model_dump(mode="json")
                            request["provider_cost_usd"] = (raw.get("usage") or {}).get("cost")
                            choice = response.choices[0]
                            request["finish_reason"] = choice.finish_reason
                            request["success"] = True
                            assistant = choice.message
                            messages.append(assistant.model_dump(mode="json", exclude_none=True))
                            transcript.append({"type": "assistant_message", "request_id": request_id,
                                               "message": assistant.model_dump(mode="json", exclude_none=True)})
                        except Exception as exc:
                            request["error"] = f"{type(exc).__name__}: {str(exc)[:400]}"
                            failures.append("provider_request_failed:" + request["error"])
                        finally:
                            request["elapsed_seconds"] = time.monotonic() - call_started
                            provider_requests.append(request)
                            transcript.append({"type": "provider_request", **request})
                        if not request["success"]:
                            break
                        if request["finish_reason"] in {"length", "max_tokens"}:
                            failures.append("output_truncation")
                            break
                        message = response.choices[0].message
                        calls = message.tool_calls or []
                        if not calls:
                            status = await call_tool("get_prediction_status", {})
                            status_body = json.loads(_content(status))
                            completed_ids = status_body.get("stored_ids", [])
                            if set(completed_ids) == set(ids):
                                break
                            failures.append("model_terminated_without_durable_completion")
                            break
                        # Reserve one call for the durable-status check after this tool turn.
                        if len(mcp_calls) + len(calls) + 1 > budgets["max_mcp_calls"]:
                            safety_ceiling_hit = True
                            failures.append("mcp_call_limit")
                            break
                        for call in calls:
                            try:
                                arguments = json.loads(call.function.arguments or "{}")
                                result = await call_tool(call.function.name, arguments)
                                result_text = _content(result)
                                messages.append({"role": "tool", "tool_call_id": call.id,
                                                 "content": result_text})
                            except Exception as exc:
                                failures.append(f"tool_execution_error:{call.function.name}:{type(exc).__name__}:{str(exc)[:300]}")
                                messages.append({"role": "tool", "tool_call_id": call.id,
                                                 "content": json.dumps({"error": failures[-1]})})
                        # Durable status is checked only after the model's requested tool calls.
                        status = await call_tool("get_prediction_status", {})
                        status_body = json.loads(_content(status))
                        completed_ids = status_body.get("stored_ids", [])
                        if set(completed_ids) == set(ids):
                            break
    except TimeoutError:
        safety_ceiling_hit = True
        failures.append("runtime_limit")
    runtime = time.monotonic() - started
    pricing_path = Path(__file__).resolve().parents[1] / "artifacts/sampo_cost_demo/pricing.json"
    pricing = json.loads(pricing_path.read_text(encoding="utf-8"))["models"][model]
    known = [r for r in provider_requests if r["usage_known"]]
    complete = all(r["usage_known"] for r in provider_requests)
    cost = None
    if complete:
        cost = sum(r["provider_cost_usd"] if r["provider_cost_usd"] is not None else
                   (max(0, r["input_tokens"]-r["cached_input_tokens"])*pricing["input_usd_per_1m"] +
                    r["cached_input_tokens"]*pricing.get("cached_input_usd_per_1m", pricing["input_usd_per_1m"]) +
                    r["output_tokens"]*pricing["output_usd_per_1m"])/1_000_000 for r in provider_requests)
    telemetry = {
        "system": "terra_single_agent", "model": model, "assigned_ids": ids,
        "completed_ids": [i for i in ids if i in set(completed_ids)],
        "failed_ids": [i for i in ids if i not in set(completed_ids)],
        "completed_examples": len(set(completed_ids) & set(ids)),
        "provider_requests": provider_requests, "provider_request_count": len(provider_requests),
        "model_calls_count": len(provider_requests), "model_call_count": len(provider_requests),
        "mcp_calls": mcp_calls, "mcp_calls_count": len(mcp_calls),
        "input_tokens": sum(r["input_tokens"] for r in known),
        "uncached_input_tokens": sum(max(0, r["input_tokens"]-r["cached_input_tokens"]) for r in known),
        "cached_input_tokens": sum(r["cached_input_tokens"] for r in known),
        "output_tokens": sum(r["output_tokens"] for r in known),
        "provider_reported_cost_usd": (sum(r["provider_cost_usd"] for r in known if r["provider_cost_usd"] is not None)
                                       if any(r["provider_cost_usd"] is not None for r in known) else None),
        "fallback_calculated_cost_usd": sum((max(0,r["input_tokens"]-r["cached_input_tokens"])*pricing["input_usd_per_1m"] + r["cached_input_tokens"]*pricing.get("cached_input_usd_per_1m",pricing["input_usd_per_1m"]) + r["output_tokens"]*pricing["output_usd_per_1m"])/1_000_000 for r in known if r["provider_cost_usd"] is None),
        "cost_usd": cost, "cost_complete": complete, "finish_reasons": [r["finish_reason"] for r in provider_requests],
        "output_truncation_count": sum(r["finish_reason"] in {"length", "max_tokens"} for r in provider_requests),
        "output_truncation_hit": any(r["finish_reason"] in {"length", "max_tokens"} for r in provider_requests),
        "runtime_seconds": runtime, "safety_ceiling_hit": safety_ceiling_hit,
        "failures": failures, "transcript_messages": len(messages), "tool_schemas": tool_schemas,
    }
    transcript.append({"type": "final_status", "stored_ids": telemetry["completed_ids"],
                       "assigned_ids": ids, "failures": failures})
    (output_dir / "telemetry.json").write_text(json.dumps(telemetry, ensure_ascii=False, indent=2)+"\n")
    with (output_dir / "transcript.jsonl").open("x", encoding="utf-8") as stream:
        for item in transcript:
            stream.write(json.dumps(item, ensure_ascii=False)+"\n")
    runtime_manifest = {
        "system": "terra_single_agent", "harness": "standalone_openai_tool_loop",
        "fedotmas_dependency": False, "model": model, "endpoint": endpoint,
        "batch_size": len(ids), "budgets": budgets,
        "mcp_server_script": str(server_script), "fresh_mcp_process": True,
        "fresh_model_conversation": True, "tools": [t["function"]["name"] for t in tool_schemas],
        "private_gt_path_provided": False, "phase5_config_exposed": False,
        "git_commit": git_commit,
    }
    (output_dir / "runtime_manifest.json").write_text(json.dumps(runtime_manifest, ensure_ascii=False, indent=2)+"\n")
    prediction_path = output_dir / ".predictions.jsonl"
    pred_rows = [json.loads(line) for line in prediction_path.read_text(encoding="utf-8").splitlines() if line] if prediction_path.exists() else []
    with (output_dir / "predictions.csv").open("x", encoding="utf-8", newline="") as stream:
        import csv
        writer = csv.DictWriter(stream, fieldnames=["example_id", "top_1", "top_2", "top_3"])
        writer.writeheader()
        writer.writerows(pred_rows)
    return telemetry
