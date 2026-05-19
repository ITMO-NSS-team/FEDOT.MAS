import argparse
import asyncio
import json
import os
import re
import socket
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm import tqdm

from fedotmas import MAW, ModelConfig
from fedotmas._settings import get_meta_model
from fedotmas.common.logging import get_logger
from fedotmas.plugins import (
    LangfusePlugin,
    LoggingPlugin,
    ToolErrorCircuitBreakerPlugin,
    ToolResultTruncationPlugin,
    WebSearchLimitPlugin,
)

from examples.gaia.data import GaiaBenchmark

load_dotenv()

RUN_ID = uuid.uuid4()
_log = get_logger("fedotmas.examples.gaia")
DEFAULT_GAIA_MCP_SERVERS = [
    "websearch-searxng",
    "web-scraping",
    "sandbox-light",
    "document",
    "media",
]
DEFAULT_GAIA_WORKER_MODEL = "openai/gpt-5-mini"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
GAIA_WEB_SCRAPING_TOOL_NAMES = {
    "goto",
    "markdown",
    "links",
    "extract",
    "evaluate",
    "screenshot",
    "status",
}
PROVIDER_ERROR_PATTERNS = (
    "Provider returned error",
    "provider_name",
    "temporarily blocked",
)


class ProviderErrorCooldown(RuntimeError):
    """Raised when the model provider asks us to back off."""


class ProviderCooldown:
    def __init__(self, seconds: int) -> None:
        self._seconds = max(0, seconds)
        self._until = 0.0
        self._reason = ""

    def activate(self, reason: str) -> None:
        if self._seconds <= 0:
            return
        self._until = max(self._until, time.monotonic() + self._seconds)
        self._reason = reason

    async def wait_if_active(self) -> None:
        remaining = self._until - time.monotonic()
        if remaining <= 0:
            return
        _log.warning(
            "Provider cooldown active for {:.0f}s after provider error: {}",
            remaining,
            self._reason[:300],
        )
        await asyncio.sleep(remaining)


def extract_solution(text: str) -> str:
    """Extract answer from <solution> tags, or return stripped text."""
    pattern = r"<solution>(.*?)</solution>"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return text.strip()


def extract_answer_from_state(state: dict[str, Any]) -> str:
    """Extract final answer from session state.

    First looks for <solution> tags in any non-query state value.
    Falls back to the last non-null, non-user_query value.
    """
    # Search all values for <solution> tags (last match wins)
    solution = None
    for key, value in state.items():
        if key == "user_query":
            continue
        if value is None:
            continue
        text = str(value)
        found = extract_solution(text)
        if found != text.strip():  # tags were found
            solution = found

    if solution is not None:
        return solution

    # Fall back: last non-null, non-user_query value
    for key in reversed(list(state.keys())):
        if key == "user_query":
            continue
        value = state[key]
        if value is not None and str(value).strip():
            return str(value).strip()

    return ""


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        _log.warning("Invalid {}={!r}; using {}", name, value, default)
        return default


def _env_list(name: str) -> list[str] | None:
    value = os.getenv(name)
    if not value:
        return None
    values = [item.strip() for item in value.split(",") if item.strip()]
    return values or None


def _gaia_mcp_servers() -> list[str] | str:
    value = os.getenv("FEDOTMAS_GAIA_MCP_SERVERS")
    if value is None:
        return DEFAULT_GAIA_MCP_SERVERS
    if value.strip().lower() == "all":
        return "all"
    return [name.strip() for name in value.split(",") if name.strip()]


def _gaia_provider_extra_body() -> dict[str, Any] | None:
    provider: dict[str, Any] = {}
    for field in ("order", "only", "ignore", "quantizations"):
        values = _env_list(f"FEDOTMAS_GAIA_PROVIDER_{field.upper()}")
        if values:
            provider[field] = values
    for field in (
        "allow_fallbacks",
        "require_parameters",
        "zdr",
        "enforce_distillable_text",
    ):
        name = f"FEDOTMAS_GAIA_PROVIDER_{field.upper()}"
        if os.getenv(name) is not None:
            provider[field] = _env_flag(name, False)
    sort_by = os.getenv("FEDOTMAS_GAIA_PROVIDER_SORT_BY")
    sort_partition = os.getenv("FEDOTMAS_GAIA_PROVIDER_SORT_PARTITION")
    if sort_by or sort_partition:
        sort: dict[str, str] = {}
        if sort_by:
            sort["by"] = sort_by
        if sort_partition:
            sort["partition"] = sort_partition
        provider["sort"] = sort
    else:
        value = os.getenv("FEDOTMAS_GAIA_PROVIDER_SORT")
        if value:
            provider["sort"] = value

    for field in ("data_collection",):
        value = os.getenv(f"FEDOTMAS_GAIA_PROVIDER_{field.upper()}")
        if value:
            provider[field] = value

    if not provider:
        return None
    return {"provider": provider}


def _gaia_meta_model() -> ModelConfig:
    return ModelConfig(
        model=os.getenv("FEDOTMAS_GAIA_META_MODEL", get_meta_model()),
        api_base=os.getenv("FEDOTMAS_GAIA_META_BASE_URL")
        or os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("FEDOTMAS_GAIA_META_API_KEY")
        or os.getenv("OPENROUTER_API_KEY")
        or os.getenv("OPENAI_API_KEY"),
        extra_body=_gaia_provider_extra_body(),
    )


def _gaia_worker_model() -> ModelConfig:
    return ModelConfig(
        model=os.getenv("FEDOTMAS_GAIA_WORKER_MODEL", DEFAULT_GAIA_WORKER_MODEL),
        api_base=os.getenv(
            "FEDOTMAS_GAIA_WORKER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL
        ),
        api_key=os.getenv("FEDOTMAS_GAIA_WORKER_API_KEY")
        or os.getenv("OPENROUTER_API_KEY")
        or os.getenv("OPENAI_API_KEY"),
        extra_body=_gaia_provider_extra_body(),
    )


def _is_provider_error(error: BaseException) -> bool:
    text = str(error).lower()
    return any(pattern.lower() in text for pattern in PROVIDER_ERROR_PATTERNS)


def _models_url(base_url: str) -> str:
    return urljoin(base_url.rstrip("/") + "/", "models")


def _check_models_endpoint(model: ModelConfig, *, timeout: float) -> None:
    if not model.api_base:
        return

    request = Request(_models_url(model.api_base))
    if model.api_key:
        request.add_header("Authorization", f"Bearer {model.api_key}")

    try:
        with urlopen(request, timeout=timeout) as response:
            if response.status >= 400:
                raise RuntimeError(
                    f"Model endpoint healthcheck failed: HTTP {response.status} "
                    f"from {_models_url(model.api_base)}"
                )
    except HTTPError as exc:
        raise RuntimeError(
            f"Model endpoint healthcheck failed: HTTP {exc.code} from "
            f"{_models_url(model.api_base)}"
        ) from exc
    except (TimeoutError, URLError, socket.gaierror, OSError) as exc:
        raise RuntimeError(
            f"Model endpoint healthcheck failed for {_models_url(model.api_base)}: "
            f"{exc}"
        ) from exc


async def preflight_model_endpoint(model: ModelConfig) -> None:
    if not _env_flag("FEDOTMAS_GAIA_PREFLIGHT_MODELS", True):
        return

    attempts = _env_int("FEDOTMAS_GAIA_PREFLIGHT_ATTEMPTS", 3)
    timeout = float(_env_int("FEDOTMAS_GAIA_PREFLIGHT_TIMEOUT", 10))
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            await asyncio.to_thread(_check_models_endpoint, model, timeout=timeout)
            return
        except Exception as exc:
            last_error = exc
            if attempt < attempts:
                await asyncio.sleep(min(2**attempt, 10))

    assert last_error is not None
    raise last_error


def build_plugins(task, enable_langfuse: bool) -> list:
    plugins = [
        LoggingPlugin(),
        ToolResultTruncationPlugin(
            max_string_chars=_env_int("FEDOTMAS_GAIA_MAX_TOOL_RESULT_CHARS", 50000),
        ),
        WebSearchLimitPlugin(
            max_calls_per_agent=_env_int("FEDOTMAS_GAIA_WEB_SEARCH_LIMIT", 4),
            hard_fail=True,
        ),
        WebSearchLimitPlugin(
            max_calls_per_agent=_env_int("FEDOTMAS_GAIA_WEB_TOOL_LIMIT", 8),
            tool_names=GAIA_WEB_SCRAPING_TOOL_NAMES,
            hard_fail=True,
            name="fedotmas_gaia_web_tool_limit",
        ),
        ToolErrorCircuitBreakerPlugin(
            max_errors_per_agent=_env_int("FEDOTMAS_GAIA_MAX_TOOL_ERRORS", 6),
            max_same_tool_error_type=_env_int(
                "FEDOTMAS_GAIA_MAX_SAME_TOOL_ERROR_TYPE", 2
            ),
        ),
    ]
    if enable_langfuse:
        plugins.append(
            LangfusePlugin(
                trace_name=f"gaia:{RUN_ID}:{task.task_id}",
                tags=["gaia", f"difficulty:{task.difficulty}"],
                metadata={
                    "task_id": task.task_id,
                    "difficulty": task.difficulty,
                    "file_name": task.file_name,
                },
            ),
        )
    return plugins


def compute_metrics_by_level(results: list) -> dict:
    difficulty_stats: dict[int, dict] = {}

    for result in results:
        difficulty = int(result["difficulty"])
        is_correct = result["is_correct"]

        if difficulty not in difficulty_stats:
            difficulty_stats[difficulty] = {"total": 0, "correct": 0}

        difficulty_stats[difficulty]["total"] += 1
        if is_correct:
            difficulty_stats[difficulty]["correct"] += 1

    metrics_by_level = {}
    overall_total = 0
    overall_correct = 0

    for difficulty in sorted(difficulty_stats.keys()):
        stats = difficulty_stats[difficulty]
        accuracy = (
            (stats["correct"] / stats["total"]) * 100 if stats["total"] > 0 else 0
        )

        metrics_by_level[f"level_{difficulty}"] = {
            "total_tasks": stats["total"],
            "correct": stats["correct"],
            "accuracy": round(accuracy, 2),
        }

        overall_total += stats["total"]
        overall_correct += stats["correct"]

    overall_accuracy = (
        (overall_correct / overall_total) * 100 if overall_total > 0 else 0
    )
    metrics_by_level["overall"] = {
        "total_tasks": overall_total,
        "correct": overall_correct,
        "accuracy": round(overall_accuracy, 2),
    }

    return metrics_by_level


def compute_token_summary(results: list) -> dict:
    total_meta_prompt = 0
    total_meta_completion = 0
    total_pipeline_prompt = 0
    total_pipeline_completion = 0

    for result in results:
        if "error" in result:
            continue
        tokens = result.get("tokens", {})
        total_meta_prompt += tokens.get("meta_prompt", 0)
        total_meta_completion += tokens.get("meta_completion", 0)
        total_pipeline_prompt += tokens.get("pipeline_prompt", 0)
        total_pipeline_completion += tokens.get("pipeline_completion", 0)

    return {
        "meta_agent": {
            "prompt_tokens": total_meta_prompt,
            "completion_tokens": total_meta_completion,
            "total_tokens": total_meta_prompt + total_meta_completion,
        },
        "pipeline": {
            "prompt_tokens": total_pipeline_prompt,
            "completion_tokens": total_pipeline_completion,
            "total_tokens": total_pipeline_prompt + total_pipeline_completion,
        },
        "grand_total": {
            "prompt_tokens": total_meta_prompt + total_pipeline_prompt,
            "completion_tokens": total_meta_completion + total_pipeline_completion,
            "total_tokens": (
                total_meta_prompt
                + total_meta_completion
                + total_pipeline_prompt
                + total_pipeline_completion
            ),
        },
    }


def print_score_by_level(metrics_by_level: dict) -> None:
    print("\n" + "=" * 50)
    print("GAIA Benchmark Results")
    print("=" * 50)

    for key in sorted(metrics_by_level.keys()):
        if key.startswith("level_"):
            level = key.split("_")[1]
            stats = metrics_by_level[key]
            print(
                f"Level {level}: {stats['accuracy']:.2f}%  ({stats['correct']}/{stats['total_tasks']})"
            )

    overall = metrics_by_level["overall"]
    print(
        f"Overall:   {overall['accuracy']:.2f}%  ({overall['correct']}/{overall['total_tasks']})"
    )
    print("=" * 50)


def print_token_summary(token_summary: dict) -> None:
    print("\n" + "=" * 50)
    print("Token Usage")
    print("=" * 50)
    meta = token_summary["meta_agent"]
    pipe = token_summary["pipeline"]
    grand = token_summary["grand_total"]
    print(
        f"Meta-agent:  {meta['total_tokens']:>10,}  (prompt: {meta['prompt_tokens']:,}, completion: {meta['completion_tokens']:,})"
    )
    print(
        f"Pipeline:    {pipe['total_tokens']:>10,}  (prompt: {pipe['prompt_tokens']:,}, completion: {pipe['completion_tokens']:,})"
    )
    print(
        f"Grand total: {grand['total_tokens']:>10,}  (prompt: {grand['prompt_tokens']:,}, completion: {grand['completion_tokens']:,})"
    )
    print("=" * 50)


@retry(
    stop=stop_after_attempt(_env_int("FEDOTMAS_GAIA_TASK_ATTEMPTS", 1)),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_not_exception_type(ProviderErrorCooldown),
)
async def process_task(
    task,
    gaia_benchmark: GaiaBenchmark,
    task_log_dir: Path,
    *,
    enable_langfuse: bool,
) -> dict:
    """Process a single GAIA task using FEDOT.MAS MAW."""
    instruction = (
        "Please encapsulate your final answer (answer ONLY) within <solution> and </solution>.\n"
        "For example: The answer to the question is <solution>42</solution>.\n\n"
    )

    query = instruction
    if task.file_path:
        query += (
            "Use available document, media, or sandbox tools to inspect the local file "
            "directly. Do not claim you cannot access it before trying an appropriate "
            "tool.\n"
        )
        query += f"File path: {task.file_path}\n"
    if task.file_name:
        query += f"File name: {task.file_name}\n"
    query += f"Question: {task.question}"

    meta_model = _gaia_meta_model()
    worker_model = _gaia_worker_model()
    await preflight_model_endpoint(worker_model)

    maw = MAW(
        meta_model=meta_model,
        mcp_servers=_gaia_mcp_servers(),
        worker_models=[worker_model],
        plugins=build_plugins(task, enable_langfuse),
        max_retries=_env_int("FEDOTMAS_GAIA_MAW_MAX_RETRIES", 1),
        two_stage=False,
    )
    try:
        state = await asyncio.wait_for(
            maw.run(query),
            timeout=_env_int("FEDOTMAS_GAIA_TASK_TIMEOUT_SECONDS", 300),
        )
    except Exception as exc:
        if _is_provider_error(exc):
            raise ProviderErrorCooldown(str(exc)) from exc
        raise

    answer = extract_answer_from_state(state)
    if not answer:
        raise ValueError("MAW produced no non-empty answer")
    is_correct = gaia_benchmark.is_correct_answer(answer, task.ground_truth)

    pipeline_result = maw.last_result

    result = {
        "task_id": task.task_id,
        "question": task.question,
        "response": answer,
        "ground_truth": task.ground_truth,
        "difficulty": task.difficulty,
        "is_correct": is_correct,
        "session_state": {k: str(v) for k, v in state.items()},
        "tokens": {
            "meta_prompt": maw.meta_prompt_tokens,
            "meta_completion": maw.meta_completion_tokens,
            "pipeline_prompt": pipeline_result.total_prompt_tokens
            if pipeline_result
            else 0,
            "pipeline_completion": pipeline_result.total_completion_tokens
            if pipeline_result
            else 0,
            "total_prompt": maw.total_prompt_tokens,
            "total_completion": maw.total_completion_tokens,
        },
        "elapsed": maw.elapsed,
    }

    # Save per-task result
    task_log_dir.mkdir(parents=True, exist_ok=True)
    with open(task_log_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)

    leaderboard = {"task_id": task.task_id, "model_answer": answer}
    with open(task_log_dir / "leaderboard.json", "w", encoding="utf-8") as f:
        json.dump(leaderboard, f, indent=2)

    return result


async def run_gaia(difficulty: str, split: str, *, enable_langfuse: bool) -> Any:
    """Run GAIA benchmark using FEDOT.MAS MAW."""
    base_log_dir = Path(__file__).resolve().parent / "gaia_logs" / f"run_{RUN_ID}"
    base_log_dir.mkdir(parents=True, exist_ok=True)

    _log.info("Logs will be saved to: {}", base_log_dir)
    _log.info("Loading GAIA benchmark (difficulty={}, split={})", difficulty, split)

    gaia = GaiaBenchmark({"difficulty": difficulty, "split": split})
    gaia.download()

    results = []
    provider_cooldown = ProviderCooldown(
        _env_int("FEDOTMAS_GAIA_PROVIDER_ERROR_COOLDOWN_SECONDS", 300)
    )

    for task in tqdm(gaia, desc="Processing GAIA tasks"):
        await provider_cooldown.wait_if_active()
        task_log_dir = base_log_dir / f"task_{task.task_id}"
        try:
            result = await process_task(
                task,
                gaia,
                task_log_dir,
                enable_langfuse=enable_langfuse,
            )
            status = "CORRECT" if result["is_correct"] else "WRONG"
            _log.info(
                "[{}] task={} answer='{}' gt='{}'",
                status,
                task.task_id,
                result["response"][:60],
                task.ground_truth,
            )
        except Exception as e:
            if isinstance(e, ProviderErrorCooldown) or _is_provider_error(e):
                provider_cooldown.activate(str(e))
            _log.error("Failed task {} after all retries: {}", task.task_id, e)
            result = {
                "task_id": task.task_id,
                "question": task.question,
                "response": "",
                "ground_truth": task.ground_truth,
                "difficulty": task.difficulty,
                "is_correct": False,
                "error": str(e),
            }
            task_log_dir.mkdir(parents=True, exist_ok=True)
            with open(task_log_dir / "result.json", "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False, default=str)
            with open(task_log_dir / "leaderboard.json", "w", encoding="utf-8") as f:
                json.dump(
                    {"task_id": task.task_id, "model_answer": ""},
                    f,
                    indent=2,
                )

        results.append(result)

    metrics_by_level = compute_metrics_by_level(results)
    token_summary = compute_token_summary(results)

    output_data = {
        "config": {
            "difficulty": difficulty,
            "split": split,
            "num_tasks": len(results),
            "langfuse_enabled": enable_langfuse,
        },
        "metrics": metrics_by_level,
        "token_summary": token_summary,
        "results": results,
    }

    with open(base_log_dir / "results.json", "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)

    print_score_by_level(metrics_by_level)
    print_token_summary(token_summary)

    return output_data


def main():
    parser = argparse.ArgumentParser(description="Run GAIA benchmark with FEDOT.MAS")
    parser.add_argument(
        "--difficulty",
        type=str,
        default="all",
        choices=["1", "2", "3", "all"],
        help="Dataset difficulty level (default: all)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation[:1]",
        help="Dataset split (default: validation[:1])",
    )
    parser.add_argument(
        "--no-langfuse",
        action="store_true",
        help="Disable Langfuse tracing for this GAIA run.",
    )
    args = parser.parse_args()

    enable_langfuse = _env_flag("GAIA_ENABLE_LANGFUSE", True) and not args.no_langfuse
    asyncio.run(
        run_gaia(
            difficulty=args.difficulty,
            split=args.split,
            enable_langfuse=enable_langfuse,
        )
    )


if __name__ == "__main__":
    main()
