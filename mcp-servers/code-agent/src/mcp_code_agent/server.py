from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urljoin

import httpx
from dotenv import load_dotenv
from fastmcp import FastMCP
from pydantic import BaseModel, Field

load_dotenv()

mcp = FastMCP("code-agent")
_log = logging.getLogger("mcp_code_agent")

OPENROUTER_URL = "https://openrouter.ai/api/v1"
OPENAI_URL = "https://api.openai.com/v1"
DEFAULT_MAX_STEPS = 5
MAX_STEPS = 8
DEFAULT_MAX_EXECUTION_SECONDS = 60
MAX_EXECUTION_SECONDS = 120
DEFAULT_MAX_OUTPUT_CHARS = 4_000
MAX_OUTPUT_CHARS = 12_000
MAX_FILES = 10
MAX_FILE_BYTES = 25 * 1024 * 1024
MAX_TOTAL_FILE_BYTES = 50 * 1024 * 1024
SUPPORTED_EXTENSIONS = {
    ".csv",
    ".tsv",
    ".xlsx",
    ".json",
    ".jsonl",
    ".txt",
    ".md",
    ".zip",
    ".pdf",
}
MAX_ANSWER_CHARS = 2_000
MAX_EVIDENCE_ITEMS = 8
MAX_EVIDENCE_CHARS = 500
MAX_ERRORS = 5
MAX_ERROR_CHARS = 300


class NestedUsage(BaseModel):
    llm_invocations: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float | None = None
    available: bool = False


class CodeAgentResult(BaseModel):
    status: Literal["completed", "incomplete", "blocked", "failed"]
    answer: str = ""
    evidence: list[str] = Field(default_factory=list)
    files_used: list[str] = Field(default_factory=list)
    steps_taken: int = 0
    errors: list[str] = Field(default_factory=list)
    error_code: str | None = None
    usage: NestedUsage = Field(default_factory=NestedUsage)
    telemetry: dict[str, int | float] = Field(default_factory=dict)


def _llm_settings() -> tuple[str, str, str] | None:
    """Resolve model and credentials without crossing provider endpoints."""
    router_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    openai_base = (os.getenv("OPENAI_BASE_URL") or OPENAI_URL).rstrip("/")
    openai_compatible_model = (
        "openai/gpt-4o-mini" if "openrouter.ai" in openai_base else "gpt-4o-mini"
    )
    settings = (
        "openai/gpt-4o-mini" if router_key else openai_compatible_model,
        router_key or openai_key,
        OPENROUTER_URL if router_key else openai_base,
    )

    for prefix in ("FEDOTMAS_GAIA_WORKER", "CODE_AGENT"):
        model = os.getenv(f"{prefix}_MODEL", "").strip()
        key = os.getenv(f"{prefix}_API_KEY", "").strip()
        base = os.getenv(f"{prefix}_BASE_URL", "").strip()
        if not any((model, key, base)):
            continue

        old_model, old_key, old_base = settings
        if base:
            endpoint = base.rstrip("/")
        elif key:
            endpoint = (
                OPENROUTER_URL if prefix == "FEDOTMAS_GAIA_WORKER" else openai_base
            ).rstrip("/")
        else:
            endpoint = old_base.rstrip("/")

        if not key:
            if endpoint == old_base.rstrip("/"):
                key = old_key
            else:
                key = {
                    OPENROUTER_URL: router_key,
                    openai_base: openai_key,
                }.get(endpoint, "")
                if not key:
                    raise ValueError(
                        f"{prefix}_API_KEY is required for {prefix}_BASE_URL"
                    )

        default_model = (
            old_model
            if endpoint == old_base.rstrip("/")
            else ("openai/gpt-4o-mini" if endpoint == OPENROUTER_URL else "gpt-4o-mini")
        )
        settings = (model or default_model, key, endpoint)

    return settings if settings[1] else None


def _secret_values(environ: Mapping[str, str] | None = None) -> tuple[str, ...]:
    source = os.environ if environ is None else environ
    return tuple(
        sorted(
            {
                value.strip()
                for name, value in source.items()
                if value
                and value.strip()
                and re.search(
                    r"API_KEY|TOKEN|SECRET|PASSWORD|ACCESS_KEY",
                    name,
                    re.IGNORECASE,
                )
            },
            key=len,
            reverse=True,
        )
    )


def _scrub(value: object, secrets: Sequence[str]) -> str:
    text = str(value or "")
    for secret in secrets:
        text = text.replace(secret, "[redacted]")
    return text


def _bounded(value: object, limit: int, secrets: Sequence[str]) -> str:
    text = _scrub(value, secrets).strip()
    if len(text) > limit:
        suffix = "… [truncated]"
        text = text[: limit - len(suffix)].rstrip() + suffix
    return text


def _result(
    *,
    status: Literal["completed", "incomplete", "blocked", "failed"],
    answer: object = "",
    evidence: Sequence[object] = (),
    files_used: Sequence[str] = (),
    steps_taken: int = 0,
    errors: Sequence[object] = (),
    error_code: str | None = None,
    usage: NestedUsage | None = None,
    started: float | None = None,
    execution_failures: int = 0,
    timeouts: int = 0,
    secrets: Sequence[str] = (),
) -> CodeAgentResult:
    duration = round(max(0.0, time.monotonic() - started), 3) if started else 0.0
    return CodeAgentResult(
        status=status,
        answer=_bounded(answer, MAX_ANSWER_CHARS, secrets),
        evidence=[
            _bounded(item, MAX_EVIDENCE_CHARS, secrets)
            for item in evidence[:MAX_EVIDENCE_ITEMS]
            if str(item or "").strip()
        ],
        files_used=[_bounded(item, 200, secrets) for item in files_used[:MAX_FILES]],
        steps_taken=steps_taken,
        errors=[
            _bounded(item, MAX_ERROR_CHARS, secrets)
            for item in errors[:MAX_ERRORS]
            if str(item or "").strip()
        ],
        error_code=error_code,
        usage=usage or NestedUsage(),
        telemetry={
            "duration_seconds": duration,
            "execution_failures": execution_failures,
            "timeouts": timeouts,
            "files_accessed": len(files_used),
        },
    )


def _load_files(files: Sequence[str]) -> list[tuple[str, str, bytes]]:
    if len(files) > MAX_FILES:
        raise ValueError("At most 10 input files are allowed")
    loaded: list[tuple[str, str, bytes]] = []
    total_size = 0
    for index, raw_path in enumerate(files):
        path = Path(raw_path).expanduser()
        try:
            if not path.is_file():
                raise FileNotFoundError
            suffix = path.suffix.lower()
            if suffix not in SUPPORTED_EXTENSIONS:
                raise ValueError(f"Unsupported file type: {suffix or '(none)'}")
            with path.open("rb") as source:
                data = source.read(MAX_FILE_BYTES + 1)
            if len(data) > MAX_FILE_BYTES:
                raise ValueError("A file exceeds the 25 MB input limit")
            total_size += len(data)
            if total_size > MAX_TOTAL_FILE_BYTES:
                raise ValueError("Input files exceed the 50 MB total limit")
        except FileNotFoundError as exc:
            raise FileNotFoundError(path.name) from exc
        sandbox_name = f"/tmp/code_agent/input_{index}{suffix}"
        loaded.append((path.name, sandbox_name, data))
    return loaded


def _usage_update(usage: NestedUsage, raw: object) -> None:
    if not isinstance(raw, dict):
        return
    token_values: dict[str, int] = {}
    for target, source in (
        ("prompt_tokens", "prompt_tokens"),
        ("completion_tokens", "completion_tokens"),
        ("total_tokens", "total_tokens"),
    ):
        value = raw.get(source)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            token_values[target] = value
    for target in ("prompt_tokens", "completion_tokens"):
        usage_value = token_values.get(target, 0)
        setattr(usage, target, getattr(usage, target) + usage_value)
    total = token_values.get(
        "total_tokens",
        token_values.get("prompt_tokens", 0) + token_values.get("completion_tokens", 0),
    )
    usage.total_tokens += total
    cost = raw.get("cost_usd", raw.get("cost"))
    if isinstance(cost, int | float) and not isinstance(cost, bool) and cost >= 0:
        usage.cost_usd = (usage.cost_usd or 0.0) + float(cost)
    if token_values or (
        isinstance(cost, int | float) and not isinstance(cost, bool) and cost >= 0
    ):
        usage.available = True


async def _request_model(
    client: httpx.AsyncClient,
    settings: tuple[str, str, str],
    messages: list[dict[str, str]],
    usage: NestedUsage,
) -> dict:
    model, api_key, base_url = settings
    usage.llm_invocations += 1
    endpoint = urljoin(base_url.rstrip("/") + "/", "chat/completions")
    body = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    if "openrouter.ai" in base_url:
        body["usage"] = {"include": True}
    response = await client.post(
        endpoint,
        headers={"Authorization": f"Bearer {api_key}"},
        json=body,
    )
    if response.is_error:
        raise RuntimeError(f"Model service returned HTTP {response.status_code}")
    payload = response.json()
    if not isinstance(payload, dict):
        raise TypeError("Model service returned an invalid response")
    _usage_update(usage, payload.get("usage"))
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("Model service returned no completion")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str):
        raise TypeError("Model service returned no JSON content")
    try:
        action = json.loads(content)
    except json.JSONDecodeError:
        start, end = content.find("{"), content.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Model response was not valid JSON") from None
        action = json.loads(content[start : end + 1])
    if not isinstance(action, dict):
        raise TypeError("Model response was not a JSON object")
    return action


def _parse_final(
    action: dict, secrets: Sequence[str]
) -> tuple[Literal["completed", "incomplete", "blocked", "failed"], str, list[str]]:
    status = action.get("status", "completed")
    if status not in {"completed", "incomplete", "blocked", "failed"}:
        status = "incomplete"
    answer = _bounded(action.get("answer", ""), MAX_ANSWER_CHARS, secrets)
    raw_evidence = action.get("evidence", [])
    if not isinstance(raw_evidence, list):
        raw_evidence = [raw_evidence]
    evidence = [
        _bounded(item, MAX_EVIDENCE_CHARS, secrets)
        for item in raw_evidence[:MAX_EVIDENCE_ITEMS]
        if isinstance(item, str) and item.strip()
    ]
    return status, answer, evidence


def _execution_output(result: object, limit: int, secrets: Sequence[str]) -> str:
    chunks: list[str] = []
    remaining = limit

    def add(label: str, value: object) -> None:
        nonlocal remaining
        if remaining <= 0:
            return
        rendered = f"{label}: {_scrub(value, secrets)}"
        if len(rendered) > remaining:
            suffix = "… [truncated]"
            if remaining <= len(suffix):
                rendered = rendered[:remaining]
            else:
                rendered = rendered[: remaining - len(suffix)].rstrip() + suffix
        chunks.append(rendered)
        remaining -= len(rendered)

    logs = getattr(result, "logs", None)
    for label in ("stdout", "stderr"):
        values = getattr(logs, label, None) if logs is not None else None
        if values:
            for value in values:
                add(label, value)
                if remaining <= 0:
                    break
    error = getattr(result, "error", None)
    if error is not None:
        add(
            "execution_error",
            str(getattr(error, "name", "PythonError"))
            + ": "
            + str(getattr(error, "value", "execution failed")),
        )
    for item in getattr(result, "results", []) or []:
        if remaining <= 0:
            break
        text = getattr(item, "text", None)
        if text:
            add("result", text)
    return "\n".join(chunks) or "Execution completed without printed output."


def _system_prompt(max_output_chars: int) -> str:
    return f"""You are a code agent for bounded computation and structured file analysis.
Files are explicitly staged under /tmp/code_agent with exact paths listed in the input. Use only those staged paths for input data. Use Python to inspect and compute; installed packages may include pandas and openpyxl, and the standard library can parse CSV, JSON, and ZIP files. Use pypdf for PDF text only if already installed. Do not use the network. Do not print large tables: print only relevant rows, columns, and concise intermediate values. Execution output is limited to {max_output_chars} characters.

If the request is mainly document reading/retrieval, return action=document. Otherwise respond with exactly one JSON object per turn:
{{"action":"execute","code":"Python source"}}
or
{{"action":"finish","status":"completed|incomplete|blocked|failed","answer":"compact result","evidence":["concise row, sheet, filename, calculation, or other evidence"]}}
or
{{"action":"document"}}

Use only the named files. Never inspect environment variables or attempt to access credentials. When an execution fails, use the error and last output to repair the code, within the step limit. Do not claim a value without evidence from the provided files or executed calculation. Return a concise answer and at most 8 evidence items."""


async def _solve(
    task: str,
    files: Sequence[str],
    context: str,
    max_steps: int,
    max_execution_seconds: float,
    max_output_chars: int,
) -> CodeAgentResult:
    started = time.monotonic()
    deadline = started + max_execution_seconds
    secrets = _secret_values()
    safe_task = _bounded(task, 8_000, secrets)
    safe_context = _bounded(context, 4_000, secrets)
    usage = NestedUsage()
    steps = 0
    execution_failures = 0
    file_names: list[str] = []
    errors: list[str] = []
    sandbox = None

    try:
        async with asyncio.timeout(max_execution_seconds):
            staged_files = await asyncio.to_thread(_load_files, files)
    except TimeoutError:
        return _result(
            status="incomplete",
            errors=["Code-agent exceeded its execution time limit"],
            error_code="CODE_AGENT_TIMEOUT",
            usage=usage,
            started=started,
            timeouts=1,
            secrets=secrets,
        )
    except FileNotFoundError as exc:
        return _result(
            status="failed",
            errors=[f"Input file not found: {exc}"],
            error_code="CODE_AGENT_FILE_NOT_FOUND",
            started=started,
            secrets=secrets,
        )
    except ValueError as exc:
        return _result(
            status="failed",
            errors=[exc],
            error_code="CODE_AGENT_INVALID_INPUT",
            started=started,
            secrets=secrets,
        )
    except OSError:
        return _result(
            status="failed",
            errors=["An input file could not be read"],
            error_code="CODE_AGENT_FILE_ACCESS",
            started=started,
            secrets=secrets,
        )
    file_names = [name for name, _, _ in staged_files]

    try:
        settings = _llm_settings()
    except ValueError as exc:
        return _result(
            status="blocked",
            files_used=file_names,
            errors=[exc],
            error_code="CODE_AGENT_MODEL_CONFIGURATION",
            started=started,
            secrets=secrets,
        )
    if settings is None:
        return _result(
            status="blocked",
            files_used=file_names,
            errors=["No model credentials are configured for code-agent"],
            error_code="CODE_AGENT_MODEL_UNAVAILABLE",
            started=started,
            secrets=secrets,
        )

    file_description = [
        {
            "name": _bounded(original, 200, secrets),
            "sandbox_path": staged,
            "bytes": len(data),
        }
        for original, staged, data in staged_files
    ]
    messages = [
        {"role": "system", "content": _system_prompt(max_output_chars)},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "task": safe_task,
                    "context": safe_context,
                    "files": file_description,
                    "max_steps": max_steps,
                    "step": 0,
                },
                ensure_ascii=False,
            ),
        },
    ]

    async def do_work() -> CodeAgentResult:
        nonlocal sandbox, steps, execution_failures
        async with httpx.AsyncClient(timeout=20) as client:
            for turn in range(max_steps + 1):
                try:
                    action = await _request_model(client, settings, messages, usage)
                except TimeoutError:
                    raise
                except Exception as exc:  # noqa: BLE001 - expose only a safe error.
                    return _result(
                        status="failed",
                        files_used=file_names,
                        steps_taken=steps,
                        errors=[
                            _bounded(
                                f"Model request failed: {type(exc).__name__}: {exc}",
                                MAX_ERROR_CHARS,
                                secrets,
                            )
                        ],
                        error_code="CODE_AGENT_MODEL_ERROR",
                        usage=usage,
                        started=started,
                        execution_failures=execution_failures,
                        secrets=secrets,
                    )
                kind = action.get("action")
                if kind == "document":
                    return _result(
                        status="blocked",
                        answer="This task is primarily document reading; use document or resource reading.",
                        files_used=file_names,
                        steps_taken=steps,
                        error_code="CODE_AGENT_DOCUMENT_READING_RECOMMENDED",
                        usage=usage,
                        started=started,
                        execution_failures=execution_failures,
                        secrets=secrets,
                    )
                if kind == "finish":
                    status, answer, evidence = _parse_final(action, secrets)
                    return _result(
                        status=status,
                        answer=answer,
                        evidence=evidence,
                        files_used=file_names,
                        steps_taken=steps,
                        errors=errors,
                        error_code=(
                            "CODE_AGENT_FAILED" if status == "failed" else None
                        ),
                        usage=usage,
                        started=started,
                        execution_failures=execution_failures,
                        secrets=secrets,
                    )
                if kind != "execute" or not isinstance(action.get("code"), str):
                    return _result(
                        status="failed",
                        answer="",
                        files_used=file_names,
                        steps_taken=steps,
                        errors=["Model returned an unsupported action"],
                        error_code="CODE_AGENT_MODEL_ERROR",
                        usage=usage,
                        started=started,
                        execution_failures=execution_failures,
                        secrets=secrets,
                    )
                if steps >= max_steps:
                    return _result(
                        status="incomplete",
                        files_used=file_names,
                        steps_taken=steps,
                        errors=[*errors, "Code execution step limit reached"],
                        error_code="CODE_AGENT_STEP_LIMIT",
                        usage=usage,
                        started=started,
                        execution_failures=execution_failures,
                        secrets=secrets,
                    )

                if sandbox is None:
                    e2b_key = os.getenv("E2B_API_KEY", "").strip()
                    if not e2b_key:
                        return _result(
                            status="blocked",
                            files_used=file_names,
                            steps_taken=steps,
                            errors=["E2B_API_KEY is required to execute code"],
                            error_code="CODE_AGENT_RUNTIME_UNAVAILABLE",
                            usage=usage,
                            started=started,
                            secrets=secrets,
                        )
                    try:
                        from e2b_code_interpreter import AsyncSandbox

                        sandbox = await AsyncSandbox.create(
                            api_key=e2b_key,
                            timeout=min(300, int(max_execution_seconds) + 30),
                            allow_internet_access=False,
                        )
                        for _, sandbox_name, data in staged_files:
                            await sandbox.files.write(sandbox_name, data)
                    except Exception as exc:  # noqa: BLE001 - sandbox setup is structured.
                        return _result(
                            status="blocked",
                            files_used=file_names,
                            steps_taken=steps,
                            errors=[
                                _bounded(
                                    f"Sandbox setup failed: {type(exc).__name__}: {exc}",
                                    MAX_ERROR_CHARS,
                                    secrets,
                                )
                            ],
                            error_code="CODE_AGENT_RUNTIME_UNAVAILABLE",
                            usage=usage,
                            started=started,
                            execution_failures=execution_failures,
                            secrets=secrets,
                        )

                steps += 1
                messages.append({"role": "assistant", "content": json.dumps(action)})
                try:
                    execution = await sandbox.run_code(
                        action["code"],
                        timeout=max(
                            1.0,
                            min(max_execution_seconds, deadline - time.monotonic()),
                        ),
                    )
                except Exception as exc:  # noqa: BLE001 - SDK errors are structured.
                    execution_failures += 1
                    safe_error = _bounded(
                        f"{type(exc).__name__}: {exc}", MAX_ERROR_CHARS, secrets
                    )
                    errors.append("Sandbox execution service failed: " + safe_error)
                    timed_out = "timeout" in type(exc).__name__.lower()
                    return _result(
                        status="incomplete" if timed_out else "failed",
                        files_used=file_names,
                        steps_taken=steps,
                        errors=errors,
                        error_code=(
                            "CODE_AGENT_TIMEOUT"
                            if timed_out
                            else "CODE_AGENT_EXECUTION_FAILED"
                        ),
                        usage=usage,
                        started=started,
                        execution_failures=execution_failures,
                        timeouts=int(timed_out),
                        secrets=secrets,
                    )

                output = _execution_output(execution, max_output_chars, secrets)
                execution_error = getattr(execution, "error", None)
                if execution_error is not None:
                    execution_failures += 1
                    execution_timed_out = (
                        "timeout" in str(getattr(execution_error, "name", "")).lower()
                    )
                    errors.append(
                        _bounded(
                            "Python execution failed: "
                            + str(getattr(execution_error, "name", "PythonError"))
                            + ": "
                            + str(
                                getattr(execution_error, "value", "execution failed")
                            ),
                            MAX_ERROR_CHARS,
                            secrets,
                        )
                    )
                    if execution_timed_out:
                        return _result(
                            status="incomplete",
                            files_used=file_names,
                            steps_taken=steps,
                            errors=errors,
                            error_code="CODE_AGENT_TIMEOUT",
                            usage=usage,
                            started=started,
                            execution_failures=execution_failures,
                            timeouts=1,
                            secrets=secrets,
                        )
                messages.append(
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "execution_result": output,
                                "execution_succeeded": execution_error is None,
                                "steps_used": steps,
                                "steps_remaining": max_steps - steps,
                            },
                            ensure_ascii=False,
                        ),
                    }
                )
            return _result(
                status="incomplete",
                files_used=file_names,
                steps_taken=steps,
                errors=[*errors, "Code execution step limit reached"],
                error_code="CODE_AGENT_STEP_LIMIT",
                usage=usage,
                started=started,
                execution_failures=execution_failures,
                secrets=secrets,
            )

    try:
        async with asyncio.timeout(max(0.001, deadline - time.monotonic())):
            return await do_work()
    except TimeoutError:
        return _result(
            status="incomplete",
            files_used=file_names,
            steps_taken=steps,
            errors=["Code-agent exceeded its execution time limit"],
            error_code="CODE_AGENT_TIMEOUT",
            usage=usage,
            started=started,
            execution_failures=execution_failures,
            timeouts=1,
            secrets=secrets,
        )
    except Exception as exc:  # noqa: BLE001 - keep failures inside the tool result.
        errors.append(
            _bounded(f"{type(exc).__name__}: {exc}", MAX_ERROR_CHARS, secrets)
        )
        return _result(
            status="failed",
            files_used=file_names,
            steps_taken=steps,
            errors=errors,
            error_code="CODE_AGENT_FAILED",
            usage=usage,
            started=started,
            execution_failures=execution_failures,
            secrets=secrets,
        )
    finally:
        if sandbox is not None:
            try:
                await sandbox.kill()
            except Exception as exc:  # noqa: BLE001 - cleanup must not mask result.
                _log.debug("Sandbox cleanup failed (%s)", type(exc).__name__)


@mcp.tool
async def solve_with_code(
    task: Annotated[str, Field(min_length=1, max_length=8_000)],
    files: Annotated[list[str], Field(max_length=MAX_FILES)] | None = None,
    context: Annotated[str, Field(max_length=4_000)] = "",
    max_steps: Annotated[int, Field(ge=1, le=MAX_STEPS)] = DEFAULT_MAX_STEPS,
    max_execution_seconds: Annotated[
        float, Field(ge=0.1, le=MAX_EXECUTION_SECONDS)
    ] = DEFAULT_MAX_EXECUTION_SECONDS,
    max_output_chars: Annotated[int, Field(ge=256, le=MAX_OUTPUT_CHARS)] = (
        DEFAULT_MAX_OUTPUT_CHARS
    ),
) -> dict:
    """Solve a bounded computation or structured-file task with iterative Python.

    Provide only the files the task needs. Python runs in a fresh E2B sandbox
    with outbound internet disabled. Use document for tasks centered on reading
    or retrieving document content rather than calculating or transforming it.
    """
    result = await _solve(
        task=task,
        files=files or [],
        context=context,
        max_steps=max_steps,
        max_execution_seconds=max_execution_seconds,
        max_output_chars=max_output_chars,
    )
    return result.model_dump()


def main() -> None:
    mcp.run(show_banner=False)
