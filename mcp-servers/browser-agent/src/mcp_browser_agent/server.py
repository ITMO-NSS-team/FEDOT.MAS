from __future__ import annotations

import inspect
import os
import re
from contextlib import suppress
from typing import Annotated, Literal
from urllib.parse import urlsplit

from fastmcp import FastMCP
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


class BrowserTaskResult(BaseModel):
    """Compact result from one bounded Browser-Use run."""

    status: Literal["completed", "incomplete", "blocked", "failed"] = Field(
        description="Whether the browser task completed, stopped early, was blocked, or failed"
    )
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


def _llm_settings() -> tuple[str, str, str | None] | None:
    """Resolve the Browser-Use LLM from the normal OpenAI-compatible environment."""
    api_key = (
        os.getenv("BROWSER_AGENT_API_KEY")
        or os.getenv("OPENROUTER_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    if not api_key:
        return None

    base_url = os.getenv("BROWSER_AGENT_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    is_openrouter = api_key == os.getenv("OPENROUTER_API_KEY")
    if not base_url and is_openrouter:
        base_url = "https://openrouter.ai/api/v1"

    default_model = "openai/gpt-4o-mini" if is_openrouter else "gpt-4o-mini"
    model = os.getenv("BROWSER_AGENT_MODEL", default_model)
    return model, api_key, base_url


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

    if findings:
        status: Literal["completed", "incomplete", "blocked", "failed"] = "completed"
    else:
        status = "incomplete" if steps_taken else "failed"
        if not errors:
            errors = ["Browser-Use stopped without returning findings"]

    return BrowserTaskResult(
        status=status,
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
            "browser type",
            "chromium",
            "playwright install",
            "browser is not installed",
        )
    )


async def _run_browser_task(task: str, max_steps: int) -> BrowserTaskResult:
    settings = _llm_settings()
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
    try:
        from browser_use import Agent, Browser, ChatOpenAI

        browser = Browser(headless=True, user_data_dir=None, keep_alive=False)
        llm = ChatOpenAI(model=model, api_key=api_key, base_url=base_url)
        agent = Agent(task=task, llm=llm, browser=browser)
        history = await agent.run(max_steps=max_steps)
        return _history_result(history)
    except Exception as exc:  # noqa: BLE001 - MCP reports execution failures as results.
        detail = _safe_text(exc, MAX_ERROR_CHARS)
        if _is_setup_error(exc):
            if "chromium" in detail.lower() or "playwright" in detail.lower():
                detail = f"{detail} Install Chromium with `just browser-use-install`."
            return BrowserTaskResult(
                status="blocked", errors=[detail or type(exc).__name__]
            )
        return BrowserTaskResult(
            status="failed",
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


@mcp.tool
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
) -> BrowserTaskResult:
    """Run an interactive or multi-step web task and return compact findings, URLs, status, and errors.

    Screenshots and internal action histories are never returned.
    """
    return await _run_browser_task(task, max_steps)


def main() -> None:
    mcp.run(show_banner=False)
