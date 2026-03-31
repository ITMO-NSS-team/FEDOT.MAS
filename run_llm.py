from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import random
import re
import sys
import urllib.request
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import load_from_disk
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
FEDOTMAS_SRC = ROOT / "packages" / "fedotmas" / "src"
if str(FEDOTMAS_SRC) not in sys.path:
    sys.path.insert(0, str(FEDOTMAS_SRC))

from maw_prompts import META_AGENT_SYSTEM_PROMPT  # noqa: E402
from all_tools_compact import COMPRESSED_CURRENT_SERVER_DESCRIPTIONS
from fedotmas.maw.models import MAWConfig as SharedMAWConfig

load_dotenv(ROOT / ".env")

if os.environ.get("OPENROUTER_HTTP_REFERER") and "OR_SITE_URL" not in os.environ:
    os.environ["OR_SITE_URL"] = os.environ["OPENROUTER_HTTP_REFERER"]
if os.environ.get("OPENROUTER_X_TITLE") and "OR_APP_NAME" not in os.environ:
    os.environ["OR_APP_NAME"] = os.environ["OPENROUTER_X_TITLE"]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_EXECUTE = True
DEFAULT_DATASET_NAME: str | None = None
DEFAULT_SEED = 42
DEFAULT_SAMPLE_BENCHMARKS = False
DEFAULT_QUESTIONS_PER_BENCHMARK = 2
DEFAULT_SHUFFLE = False
DEFAULT_LIMIT: int | None = None
DEFAULT_MAX_QUERIES: int | None = None
DEFAULT_MAX_RETRIES = 2
DEFAULT_MODELS_URL = "https://openrouter.ai/api/v1/models"
DEFAULT_DATASET_PATH = ROOT / "data" / "benchmark_queries"
DEFAULT_OUTPUT_DIR = ROOT / "maw_runs"
APP_NAME = "maw_cost_estimator_schema"
OUTPUT_KEY = "maw_config"
MODELS_WITHOUT_OUTPUT_SCHEMA = {
    "openai/gpt-5.4",
    "anthropic/claude-sonnet-4.6",
}


@dataclass(frozen=True)
class RunConfig:
    model_id: str
    execute: bool
    dataset_name: str | None
    seed: int
    sample_benchmarks: bool
    questions_per_benchmark: int
    shuffle: bool
    limit: int | None
    max_queries: int | None
    max_retries: int
    models_url: str
    dataset_path: Path
    output_path: Path
    write_csv: bool
    summary_csv_path: Path | None
    requests_csv_path: Path | None

# ---------------------------------------------------------------------------
# Schema — recursive fedotmas models for ADK structured output
# ---------------------------------------------------------------------------


def to_plain_data(value: Any) -> Any:
    """Recursively convert ADK / Pydantic objects into plain Python data."""
    if isinstance(value, BaseModel):
        return {
            key: to_plain_data(val)
            for key, val in value.model_dump(mode="python").items()
        }
    if isinstance(value, dict):
        return {key: to_plain_data(val) for key, val in value.items()}
    if isinstance(value, list):
        return [to_plain_data(item) for item in value]
    return value


def normalize_child(child: Any) -> dict[str, Any]:
    if isinstance(child, dict):
        return normalize_pipeline(child)
    if isinstance(child, str):
        parsed = extract_json_payload(child)
        if isinstance(parsed, dict):
            return normalize_pipeline(parsed)
    raise TypeError(f"Invalid pipeline child: expected object, got {type(child).__name__}")


def normalize_pipeline(node: dict[str, Any]) -> dict[str, Any]:
    node_type = node.get("type")
    agent_name = node.get("agent_name")
    children = node.get("children")
    has_children = isinstance(children, list) and len(children) > 0

    # Some models mislabel leaf agent nodes as sequential/parallel. Repair only
    # the unambiguous case where there are no children but an agent ref exists.
    if agent_name and not has_children:
        return {"type": "agent", "agent_name": agent_name}

    if node_type is None and has_children:
        node_type = "sequential"

    normalized: dict[str, Any] = {"type": node_type}

    if node_type == "agent":
        if agent_name is not None:
            normalized["agent_name"] = agent_name
        return normalized

    if node_type == "loop" and "max_iterations" in node:
        normalized["max_iterations"] = node["max_iterations"]

    if node_type in {"sequential", "parallel", "loop"}:
        normalized["children"] = [normalize_child(child) for child in (children or [])]

    return normalized


def normalize_result(result: dict[str, Any]) -> dict[str, Any]:
    normalized = to_plain_data(result)
    if not isinstance(normalized, dict):
        raise TypeError(
            f"Expected planner result to be an object, got {type(normalized).__name__}"
        )
    pipeline = normalized.get("pipeline")
    if isinstance(pipeline, dict):
        normalized["pipeline"] = normalize_pipeline(pipeline)
    return normalized


def extract_response_text(events: list[Any]) -> str | None:
    for event in reversed(events):
        content = getattr(event, "content", None)
        parts = getattr(content, "parts", None) or []
        texts = [part.text for part in parts if getattr(part, "text", None)]
        if texts:
            return "".join(texts)
    return None


def extract_json_payload(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    decoder = json.JSONDecoder()
    starts = [index for index, char in enumerate(cleaned) if char == "{"] or [0]
    for start in starts:
        try:
            payload, _ = decoder.raw_decode(cleaned[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload

    return json.loads(cleaned)


def build_retry_query(query: str, error: str) -> str:
    return (
        f"{query}\n\n"
        f"PREVIOUS ATTEMPT FAILED: {error}\n"
        f"Please fix this error in your response."
    )


def validate_maw_config(result: dict[str, Any]) -> SharedMAWConfig:
    return SharedMAWConfig.model_validate(normalize_result(result))


def slugify_model_id(model_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", model_id).strip("_")


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(
        description="Run ADK + recursive MAWConfig generation for one explicit model."
    )
    parser.add_argument(
        "model_id",
        help="OpenRouter model id to use for this run, for example openai/gpt-5.4-mini",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare dataset selection and output files without sending requests.",
    )
    parser.add_argument(
        "--dataset-name",
        default=DEFAULT_DATASET_NAME,
        help="Optional dataset_name filter.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Seed used only when --shuffle or --sample-benchmarks is enabled.",
    )
    parser.add_argument(
        "--sample-benchmarks",
        action="store_true",
        default=DEFAULT_SAMPLE_BENCHMARKS,
        help="Sample a fixed number of questions per benchmark instead of using all rows.",
    )
    parser.add_argument(
        "--questions-per-benchmark",
        type=int,
        default=DEFAULT_QUESTIONS_PER_BENCHMARK,
        help="How many rows to keep per benchmark when --sample-benchmarks is enabled.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        default=DEFAULT_SHUFFLE,
        help="Shuffle selected rows before applying --limit/--max-queries.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Optional cap after filtering/sampling and before --max-queries.",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=DEFAULT_MAX_QUERIES,
        help="Optional final cap on the number of processed rows.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="Retries per request before fallback or failure.",
    )
    parser.add_argument(
        "--models-url",
        default=DEFAULT_MODELS_URL,
        help="OpenRouter models endpoint used for pricing lookup.",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to the benchmark dataset saved with datasets.load_from_disk.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Path to the JSON report. Defaults to maw_runs/run_llm_<model>.json",
    )
    parser.add_argument(
        "--write-csv",
        action="store_true",
        help="Also write CSV reports.",
    )
    parser.add_argument(
        "--summary-csv-path",
        type=Path,
        default=None,
        help="Path to the per-model summary CSV. Used only with --write-csv.",
    )
    parser.add_argument(
        "--requests-csv-path",
        type=Path,
        default=None,
        help="Path to the per-request CSV. Used only with --write-csv.",
    )
    args = parser.parse_args()
    model_slug = slugify_model_id(args.model_id)
    output_path = args.output_path or (DEFAULT_OUTPUT_DIR / f"run_llm_{model_slug}.json")
    summary_csv_path = None
    requests_csv_path = None
    if args.write_csv:
        summary_csv_path = args.summary_csv_path or (
            DEFAULT_OUTPUT_DIR / f"run_llm_{model_slug}_summary.csv"
        )
        requests_csv_path = args.requests_csv_path or (
            DEFAULT_OUTPUT_DIR / f"run_llm_{model_slug}_requests.csv"
        )

    return RunConfig(
        model_id=args.model_id,
        execute=not args.dry_run and DEFAULT_EXECUTE,
        dataset_name=args.dataset_name,
        seed=args.seed,
        sample_benchmarks=args.sample_benchmarks,
        questions_per_benchmark=args.questions_per_benchmark,
        shuffle=args.shuffle,
        limit=args.limit,
        max_queries=args.max_queries,
        max_retries=args.max_retries,
        models_url=args.models_url,
        dataset_path=args.dataset_path,
        output_path=output_path,
        write_csv=args.write_csv,
        summary_csv_path=summary_csv_path,
        requests_csv_path=requests_csv_path,
    )

# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def load_rows(config: RunConfig) -> list[dict[str, Any]]:
    rows = list(load_from_disk(str(config.dataset_path)))
    if config.dataset_name:
        rows = [row for row in rows if row["dataset_name"] == config.dataset_name]

    if config.sample_benchmarks:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in tqdm(rows, desc="Grouping dataset rows", leave=False):
            grouped.setdefault(row["dataset_name"], []).append(row)

        sampled: list[dict[str, Any]] = []
        for dataset_name in tqdm(
            sorted(grouped), desc="Sampling benchmarks", leave=False
        ):
            bucket = list(grouped[dataset_name])
            rng = random.Random(f"{config.seed}:{dataset_name}")
            rng.shuffle(bucket)
            sampled.extend(bucket[: config.questions_per_benchmark])
        result = sampled
    else:
        result = list(rows)

    if config.shuffle:
        random.Random(config.seed).shuffle(result)

    if config.limit is not None:
        result = result[: config.limit]
    if config.max_queries is not None:
        result = result[: config.max_queries]
    return result


def print_selection(rows: list[dict[str, Any]]) -> None:
    counts = Counter(row["dataset_name"] for row in rows)
    print("Selected questions:")
    for name, count in sorted(counts.items()):
        print(f"  {name}: {count}")
    print(f"  total: {len(rows)}")


# ---------------------------------------------------------------------------
# OpenRouter model pricing
# ---------------------------------------------------------------------------


def fetch_model_spec(config: RunConfig) -> dict[str, Any]:
    data = json.loads(
        urllib.request.urlopen(config.models_url, timeout=30).read().decode("utf-8")
    )["data"]
    model = next((item for item in data if item["id"] == config.model_id), None)
    if model is None:
        raise RuntimeError(f"Model not found in OpenRouter /models: {config.model_id}")
    pricing = model.get("pricing") or {}
    return {
        "id": config.model_id,
        "litellm_model": f"openrouter/{config.model_id}",
        "input_price_per_token": float(pricing.get("prompt") or 0),
        "output_price_per_token": float(pricing.get("completion") or 0),
    }


def estimate_cost(model_spec: dict[str, Any], usage: dict[str, int]) -> float:
    return (
        usage.get("prompt_tokens", 0) * model_spec["input_price_per_token"]
        + usage.get("completion_tokens", 0) * model_spec["output_price_per_token"]
    )


# ---------------------------------------------------------------------------
# Prompt / agent helpers
# ---------------------------------------------------------------------------


def build_system_prompt(model_id: str) -> str:
    return META_AGENT_SYSTEM_PROMPT.substitute(
        mcp_servers_desc=json.dumps(
            COMPRESSED_CURRENT_SERVER_DESCRIPTIONS, ensure_ascii=False, indent=2
        ),
        available_models=f'- "{model_id}"',
    )


def make_agent_name(model_id: str) -> str:
    return f"planner_{re.sub(r'[^a-zA-Z0-9]+', '_', model_id).strip('_')}"


# ---------------------------------------------------------------------------
# Planner — ADK-based structured output, following _adk_runner.py patterns
# ---------------------------------------------------------------------------


class Planner:
    """Wraps a single LlmAgent + Runner for a given OpenRouter model."""

    def __init__(self, model_spec: dict[str, Any], config: RunConfig) -> None:
        from google.adk.sessions import InMemorySessionService

        self.model_spec = model_spec
        self.config = config
        self.sessions = InMemorySessionService()

    def _make_agent(self, use_output_schema: bool) -> LlmAgent:
        from google.adk.agents import LlmAgent
        from google.adk.models.lite_llm import LiteLlm

        agent_kwargs = dict(
            name=make_agent_name(self.model_spec["id"]),
            model=LiteLlm(model=self.model_spec["litellm_model"]),
            instruction=build_system_prompt(self.model_spec["id"]),
            output_key=OUTPUT_KEY,
        )
        if use_output_schema:
            agent_kwargs["output_schema"] = SharedMAWConfig
        return LlmAgent(**agent_kwargs)

    async def run(
        self,
        query: str,
        *,
        force_output_schema: bool | None = None,
        extra_feedback: str | None = None,
    ) -> tuple[dict[str, Any], dict[str, int], str | None, str]:
        """Send *query* and return (structured_result, usage_dict).

        Retries up to max_retries times on errors, appending the error
        message to the user prompt on each retry (following production
        _adk_runner.py pattern).
        """
        last_error: Exception | None = None
        effective_query = query
        if extra_feedback:
            effective_query = build_retry_query(effective_query, extra_feedback)

        if force_output_schema is None:
            use_output_schema = self.model_spec["id"] not in MODELS_WITHOUT_OUTPUT_SCHEMA
        else:
            use_output_schema = force_output_schema

        for attempt in range(self.config.max_retries + 1):
            try:
                return await self._execute(effective_query, use_output_schema)
            except Exception as exc:
                last_error = exc
                if attempt < self.config.max_retries:
                    tqdm.write(
                        f"  [retry {attempt + 1}/{self.config.max_retries}] "
                        f"{self.model_spec['id']}: {exc}"
                    )
                    effective_query = build_retry_query(query, str(exc))
                    await asyncio.sleep(2**attempt)

        if use_output_schema:
            tqdm.write(
                f"  FALLBACK [{self.model_spec['id']}]: "
                "recursive structured output failed, retrying as text_json"
            )
            return await self.run(
                query,
                force_output_schema=False,
                extra_feedback=str(last_error),
            )

        raise last_error  # type: ignore[misc]

    async def _execute(
        self, query: str, use_output_schema: bool
    ) -> tuple[dict[str, Any], dict[str, int], str | None, str]:
        from google.adk.runners import Runner
        from google.genai import types

        session_id = uuid.uuid4().hex
        await self.sessions.create_session(
            app_name=APP_NAME,
            user_id="benchmark",
            session_id=session_id,
            state={},
        )

        message = types.Content(
            role="user",
            parts=[types.Part.from_text(text=query)],
        )

        total_prompt = 0
        total_completion = 0
        events = []
        agent = self._make_agent(use_output_schema)

        async with Runner(
            app_name=APP_NAME,
            agent=agent,
            session_service=self.sessions,
        ) as runner:
            async for event in runner.run_async(
                user_id="benchmark",
                session_id=session_id,
                new_message=message,
            ):
                events.append(event)
                if event.partial:
                    continue

                if event.usage_metadata:
                    um = event.usage_metadata
                    total_prompt += um.prompt_token_count or 0
                    total_completion += um.candidates_token_count or 0

                if event.error_code:
                    raise RuntimeError(
                        f"LLM error {event.error_code}: {event.error_message}"
                    )

        final_session = await self.sessions.get_session(
            app_name=APP_NAME,
            user_id="benchmark",
            session_id=session_id,
        )
        if final_session is None:
            raise RuntimeError("Session lost after execution")

        response_text: str | None = None
        response_format = "structured_output" if use_output_schema else "text_json"

        if use_output_schema:
            raw_output = final_session.state.get(OUTPUT_KEY)
            if raw_output is None:
                raise RuntimeError(
                    f"Agent did not produce '{OUTPUT_KEY}' in session state"
                )
        else:
            response_text = extract_response_text(events)
            if response_text is None:
                raise RuntimeError("No text response found in ADK events")
            raw_output = extract_json_payload(response_text)

        usage = {
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
        }
        return raw_output, usage, response_text, response_format


async def execute_query(
    planner: Planner,
    query: str,
    model_id: str,
    request_index: int,
) -> tuple[SharedMAWConfig, dict[str, int], str, str]:
    result, usage, raw_response_text, response_format = await planner.run(query)

    try:
        config = validate_maw_config(result)
    except ValidationError as exc:
        if response_format != "structured_output":
            raise

        tqdm.write(
            f"  FALLBACK [{model_id}] q{request_index}: "
            "structured output invalid, retrying as text_json"
        )
        result, fallback_usage, raw_response_text, response_format = await planner.run(
            query,
            force_output_schema=False,
            extra_feedback=str(exc),
        )
        config = validate_maw_config(result)
        usage = {
            "prompt_tokens": usage.get("prompt_tokens", 0)
            + fallback_usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0)
            + fallback_usage.get("completion_tokens", 0),
        }

    return config, usage, raw_response_text or "", response_format


# ---------------------------------------------------------------------------
# Report writing
# ---------------------------------------------------------------------------


def write_outputs(report: dict[str, Any], config: RunConfig) -> None:
    config.output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if not config.write_csv:
        return

    assert config.summary_csv_path is not None
    assert config.requests_csv_path is not None

    with config.summary_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model_id",
                "requests",
                "responses",
                "valid_configs",
                "total_prompt_tokens",
                "total_completion_tokens",
                "total_cost_usd",
                "avg_cost_usd",
            ],
        )
        writer.writeheader()
        for model_report in report["models"]:
            requests = model_report["requests"]
            writer.writerow(
                {
                    "model_id": model_report["model_id"],
                    "requests": len(requests),
                    "responses": sum(1 for r in requests if r.get("config_valid")),
                    "valid_configs": sum(
                        1 for r in requests if r.get("config_valid")
                    ),
                    "total_prompt_tokens": sum(
                        (r.get("usage") or {}).get("prompt_tokens", 0) for r in requests
                    ),
                    "total_completion_tokens": sum(
                        (r.get("usage") or {}).get("completion_tokens", 0)
                        for r in requests
                    ),
                    "total_cost_usd": model_report["estimated_total_cost_usd"],
                    "avg_cost_usd": (
                        model_report["estimated_total_cost_usd"] / len(requests)
                        if requests
                        else 0
                    ),
                }
            )

    with config.requests_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model_id",
                "request_index",
                "dataset_name",
                "query",
                "has_response",
                "has_error",
                "error_type",
                "prompt_tokens",
                "completion_tokens",
                "estimated_cost_usd",
                "completion_text",
                "normalized_completion_text",
                "response_format",
                "config_valid",
                "agent_count",
                "pipeline_type",
                "worker_models",
                "tools",
            ],
        )
        writer.writeheader()
        for model_report in report["models"]:
            for item in model_report["requests"]:
                config = item.get("maw_config") or {}
                agents = config.get("agents") or []
                writer.writerow(
                    {
                        "model_id": model_report["model_id"],
                        "request_index": item["request_index"],
                        "dataset_name": item["dataset_name"],
                        "query": item["query"],
                        "has_response": bool(
                            item.get("raw_response_text") or item.get("config_valid")
                        ),
                        "has_error": "error" in item,
                        "error_type": (item.get("error") or {}).get("type", ""),
                        "prompt_tokens": (item.get("usage") or {}).get(
                            "prompt_tokens", 0
                        ),
                        "completion_tokens": (item.get("usage") or {}).get(
                            "completion_tokens", 0
                        ),
                        "estimated_cost_usd": item.get("estimated_cost_usd", 0),
                        "completion_text": item.get("raw_response_text", ""),
                        "normalized_completion_text": item.get(
                            "normalized_response_text", ""
                        ),
                        "response_format": item.get("response_format", ""),
                        "config_valid": item.get("config_valid", False),
                        "agent_count": len(agents),
                        "pipeline_type": (config.get("pipeline") or {}).get("type", ""),
                        "worker_models": "|".join(
                            sorted(
                                {a.get("model", "") for a in agents if a.get("model")}
                            )
                        ),
                        "tools": "|".join(
                            sorted(
                                {tool for a in agents for tool in a.get("tools", [])}
                            )
                        ),
                    }
                )


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


async def run(config: RunConfig) -> None:
    if config.execute and "OPENROUTER_API_KEY" not in os.environ:
        raise RuntimeError("OPENROUTER_API_KEY not found in .env")

    rows = load_rows(config)
    print_selection(rows)
    model_spec = fetch_model_spec(config)

    report: dict[str, Any] = {
        "mode": "execute" if config.execute else "dry_run",
        "runtime": "google-adk",
        "models": [],
    }

    try:
        planner = Planner(model_spec, config) if config.execute else None
        model_report: dict[str, Any] = {
            "model_id": model_spec["id"],
            "estimated_total_cost_usd": 0.0,
            "requests": [],
        }
        report["models"].append(model_report)

        for index, row in enumerate(
            tqdm(rows, desc=model_spec["id"], leave=False), start=1
        ):
            item: dict[str, Any] = {
                "request_index": index,
                "dataset_name": row["dataset_name"],
                "query": row["query"],
            }

            if config.execute:
                assert planner is not None
                try:
                    maw_config, usage, raw_response_text, response_format = await execute_query(
                        planner,
                        row["query"],
                        model_spec["id"],
                        index,
                    )
                except Exception as exc:
                    tqdm.write(
                        f"  ERROR [{model_spec['id']}] "
                        f"q{index}: {type(exc).__name__}: {exc}"
                    )
                    item["error"] = {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
                else:
                    item["maw_config"] = maw_config.model_dump(mode="json")
                    item["raw_response_text"] = raw_response_text
                    item["normalized_response_text"] = json.dumps(
                        item["maw_config"], ensure_ascii=False, indent=2
                    )
                    item["response_format"] = response_format
                    item["usage"] = usage
                    item["estimated_cost_usd"] = estimate_cost(model_spec, usage)
                    item["config_valid"] = True
                    model_report["estimated_total_cost_usd"] += item[
                        "estimated_cost_usd"
                    ]
            else:
                item["note"] = "Dry run only. No request sent."

            model_report["requests"].append(item)
            write_outputs(report, config)
    except KeyboardInterrupt:
        write_outputs(report, config)
        raise

    write_outputs(report, config)


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
