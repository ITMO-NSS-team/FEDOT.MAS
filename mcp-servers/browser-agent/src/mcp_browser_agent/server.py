from __future__ import annotations

import inspect
import os
import re
from contextlib import suppress
from typing import Annotated, Literal
from urllib.parse import urlsplit

from fastmcp import FastMCP
from fastmcp.tools import ToolResult
from pydantic import BaseModel, Field

MAX_FINDINGS_CHARS = 8_000
MAX_URLS = 20
MAX_URL_CHARS = 1_000
MAX_ERRORS = 5
MAX_ERROR_CHARS = 500
MAX_STEPS = 25

_IMAGE_DATA_URI = re.compile(
    r"data:image/[^;]+;base64,[A-Za-z0-9+/=\s]+", re.IGNORECASE
)
_SCREENSHOT_FIELD = re.compile(
    r'("?(?:screenshot|image|base64)"?\s*:\s*")([^"\\]*(?:\\.[^"\\]*)*)(")',
    re.IGNORECASE,
)
_LONG_BASE64 = re.compile(
    r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{512,}={0,2}(?![A-Za-z0-9+/])"
)

mcp = FastMCP("browser-agent")


class BrowserUsage(BaseModel):
    """Provider-reported usage; null means unavailable, not free."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    llm_invocations: int | None = Field(
        default=None,
        description="Attempted Browser-Use LLM invocations, including failed calls",
    )


class BrowserTaskResult(BaseModel):
    """Compact result from one bounded Browser-Use run."""

    status: Literal["completed", "incomplete", "blocked", "failed"] = Field(
        description="Whether the browser task completed, stopped early, was blocked, or failed"
    )
    usage: BrowserUsage = Field(default_factory=BrowserUsage)
    findings: str = Field(
        default="", description="Concise final findings from the task"
    )
    relevant_urls: list[str] = Field(
        default_factory=list, description="Relevant HTTP or HTTPS URLs visited"
    )
    steps_taken: int = Field(default=0, description="Number of browser steps executed")
    duration_seconds: float | None = Field(
        default=None, description="Browser task duration in seconds"
    )
    errors: list[str] = Field(
        default_factory=list, description="Short errors or blockers"
    )


OPENROUTER_URL = "https://openrouter.ai/api/v1"
OPENAI_URL = "https://api.openai.com/v1"


def _llm_settings() -> tuple[str, str, str] | None:
    """Select an endpoint/key pair before applying scoped model overrides.

    A different endpoint must supply its own key. Model-only overrides can
    inherit the selected provider; keys are never borrowed across endpoints.
    """
    router_key = os.getenv("OPENROUTER_API_KEY")
    if router_key:
        settings = ("openai/gpt-4o-mini", router_key, OPENROUTER_URL)
    else:
        settings = (
            "gpt-4o-mini",
            os.getenv("OPENAI_API_KEY", ""),
            os.getenv("OPENAI_BASE_URL") or OPENAI_URL,
        )

    for prefix in ("FEDOTMAS_GAIA_WORKER", "BROWSER_AGENT"):
        model = os.getenv(f"{prefix}_MODEL")
        key = os.getenv(f"{prefix}_API_KEY")
        base = os.getenv(f"{prefix}_BASE_URL")
        if not any((model, key, base)):
            continue
        old_model, old_key, old_base = settings
        if base:
            endpoint = base.rstrip("/")
        elif key:
            endpoint = (
                OPENROUTER_URL
                if prefix == "FEDOTMAS_GAIA_WORKER"
                else os.getenv("OPENAI_BASE_URL") or OPENAI_URL
            ).rstrip("/")
        else:
            endpoint = old_base.rstrip("/")
        if not key:
            if endpoint == old_base.rstrip("/"):
                key = old_key
            else:
                # Only reuse standard credentials for their own endpoint.
                key = {
                    OPENROUTER_URL: router_key,
                    (os.getenv("OPENAI_BASE_URL") or OPENAI_URL).rstrip("/"): os.getenv(
                        "OPENAI_API_KEY"
                    ),
                }.get(endpoint)
                if not key:
                    raise ValueError(
                        f"{prefix}_API_KEY is required for {prefix}_BASE_URL"
                    )
        default_model = (
            old_model
            if endpoint == old_base.rstrip("/")
            else ("openai/gpt-4o-mini" if endpoint == OPENROUTER_URL else "gpt-4o-mini")
        )
        settings = (model or default_model, key or "", endpoint)
    return settings if settings[1] else None


def _usage_result(summary: object) -> BrowserUsage:
    def count(name: str) -> int | None:
        value = (
            summary.get(name)
            if isinstance(summary, dict)
            else getattr(summary, name, None)
        )
        return (
            value
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0
            else None
        )

    return BrowserUsage(
        prompt_tokens=count("total_prompt_tokens"),
        completion_tokens=count("total_completion_tokens"),
        total_tokens=count("total_tokens"),
        llm_invocations=count("entry_count"),
    )


def _track_llm_calls(agent: object) -> list[int] | None:
    """Count all registered LLM calls, including responses without token usage."""
    service = getattr(agent, "token_cost_service", None)
    models = getattr(service, "registered_llms", None)
    if not isinstance(models, dict):
        return None
    counter = [0]

    def tracked(original):
        async def invoke(*args, **kwargs):
            counter[0] += 1
            return await original(*args, **kwargs)

        return invoke

    seen = set()
    for llm in models.values():
        if id(llm) not in seen:
            object.__setattr__(llm, "ainvoke", tracked(llm.ainvoke))
            seen.add(id(llm))
    return counter


def _partial_usage(agent: object) -> BrowserUsage:
    """Recover recorded tokens on exceptions without fetching pricing again."""
    service = getattr(agent, "token_cost_service", None)
    entries = getattr(service, "usage_history", None)
    if not isinstance(entries, list):
        return BrowserUsage()
    prompt = sum(entry.usage.prompt_tokens for entry in entries)
    completion = sum(entry.usage.completion_tokens for entry in entries)
    return BrowserUsage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
        llm_invocations=len(entries),
    )


def _safe_text(value: object, limit: int) -> str:
    text = str(value or "")
    text = _IMAGE_DATA_URI.sub("[image data omitted]", text)
    text = _SCREENSHOT_FIELD.sub(r"\1[image data omitted]\3", text)
    text = _LONG_BASE64.sub("[binary data omitted]", text)
    text = text.strip()
    if len(text) > limit:
        text = text[:limit].rstrip() + "… [truncated]"
    return text


def _history_value(history: object, name: str, default: object = None) -> object:
    value = getattr(history, name, default)
    return value() if callable(value) else value


def _history_result(history: object) -> BrowserTaskResult:
    if history is None:
        return BrowserTaskResult(
            status="failed", errors=["Browser-Use returned no task history"]
        )

    raw_findings = _history_value(history, "final_result", "")
    findings = _safe_text(raw_findings, MAX_FINDINGS_CHARS)

    raw_urls = _history_value(history, "urls", []) or []
    if isinstance(raw_urls, str):
        raw_urls = [raw_urls]
    urls: list[str] = []
    for value in raw_urls:
        url = _safe_text(value, MAX_URL_CHARS)
        try:
            parsed = urlsplit(url)
        except ValueError:
            continue
        if parsed.scheme in {"http", "https"} and parsed.netloc:
            urls.append(url)
        if len(urls) >= MAX_URLS:
            break

    raw_errors = _history_value(history, "errors", []) or []
    if isinstance(raw_errors, str):
        raw_errors = [raw_errors]
    errors = [
        _safe_text(error, MAX_ERROR_CHARS)
        for error in raw_errors
        if error is not None and _safe_text(error, MAX_ERROR_CHARS)
    ][:MAX_ERRORS]

    try:
        steps_taken = len(history)  # type: ignore[arg-type]
    except (TypeError, AttributeError):
        steps_taken = 0

    raw_duration = _history_value(history, "total_duration_seconds")
    try:
        duration = float(raw_duration) if raw_duration is not None else None
    except (TypeError, ValueError):
        duration = None

    done = _history_value(history, "is_done", False) is True
    successful = _history_value(history, "is_successful")
    if done and successful is True:
        status = "completed"
    elif done and successful is False:
        status = "failed"
    else:
        status = "incomplete"
    if status != "completed" and not errors:
        errors = [
            "Browser-Use reported task failure"
            if status == "failed"
            else "Browser-Use stopped without confirmed completion"
        ]

    return BrowserTaskResult(
        status=status,
        usage=_usage_result(getattr(history, "usage", None)),
        findings=findings,
        relevant_urls=urls,
        steps_taken=steps_taken,
        duration_seconds=duration,
        errors=errors,
    )


def _is_setup_error(error: Exception) -> bool:
    message = str(error).lower()
    return isinstance(error, ModuleNotFoundError) or any(
        marker in message
        for marker in (
            "executable doesn't exist",
            "no chrome/chromium executable found",
            "no local chrome/chromium install found",
            "no browser executable found",
            "playwright install",
            "browser is not installed",
        )
    )


async def _run_browser_task(task: str, max_steps: int) -> BrowserTaskResult:
    try:
        settings = _llm_settings()
    except ValueError as exc:
        return BrowserTaskResult(status="blocked", errors=[str(exc)])
    if settings is None:
        return BrowserTaskResult(
            status="blocked",
            errors=[
                (
                    "Set OPENAI_API_KEY or OPENROUTER_API_KEY (or BROWSER_AGENT_API_KEY) "
                    "to run the Browser-Use LLM loop."
                )
            ],
        )

    model, api_key, base_url = settings
    browser = None
    agent = None
    llm_calls = None
    try:
        from browser_use import Agent, Browser, ChatOpenAI

        browser = Browser(headless=True, user_data_dir=None, keep_alive=False)
        llm = ChatOpenAI(model=model, api_key=api_key, base_url=base_url)
        agent = Agent(task=task, llm=llm, browser=browser, calculate_cost=True)
        llm_calls = _track_llm_calls(agent)
        history = await agent.run(max_steps=max_steps)
        result = _history_result(history)
        if result.usage.total_tokens is None:
            result.usage = _partial_usage(agent)
        if llm_calls is not None:
            result.usage.llm_invocations = llm_calls[0]
        return result
    except Exception as exc:  # noqa: BLE001 - MCP reports execution failures as results.
        usage = _partial_usage(agent)
        if llm_calls is not None:
            usage.llm_invocations = llm_calls[0]
        detail = _safe_text(exc, MAX_ERROR_CHARS)
        if _is_setup_error(exc):
            if "chromium" in detail.lower() or "playwright" in detail.lower():
                detail = f"{detail} Install Chromium with `just browser-use-install`."
            return BrowserTaskResult(
                status="blocked", errors=[detail or type(exc).__name__], usage=usage
            )
        return BrowserTaskResult(
            status="failed",
            usage=usage,
            errors=[f"Browser task failed: {detail or type(exc).__name__}"],
        )
    finally:
        if browser is not None:
            close = getattr(browser, "kill", None) or getattr(browser, "stop", None)
            if callable(close):
                with suppress(Exception):
                    outcome = close()
                    if inspect.isawaitable(outcome):
                        await outcome


@mcp.tool(output_schema=BrowserTaskResult.model_json_schema())
async def complete_browser_task(
    task: Annotated[
        str,
        Field(
            min_length=1,
            max_length=8_000,
            description="Web task for Browser-Use to complete",
        ),
    ],
    max_steps: Annotated[
        int,
        Field(
            ge=1, le=MAX_STEPS, description=f"Maximum browser steps (1 to {MAX_STEPS})"
        ),
    ] = 12,
) -> ToolResult:
    """Run an interactive or multi-step web task and return compact findings, URLs, status, and errors.

    Screenshots and internal action histories are never returned.
    """
    result = await _run_browser_task(task, max_steps)
    payload = result.model_dump(mode="json")
    failed = result.status != "completed"
    if failed:
        payload["error_code"] = f"BROWSER_AGENT_{result.status.upper()}"
    return ToolResult(
        structured_content=payload,
        is_error=failed,
        meta={"error_code": payload["error_code"]} if failed else None,
    )


def main() -> None:
    mcp.run(show_banner=False)
