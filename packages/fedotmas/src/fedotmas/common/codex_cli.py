"""Host-native Codex CLI transport for an authenticated Codex subscription.

Models whose names start with ``host/`` or ``codex/`` are executed by the
locally installed ``codex`` command.  This is deliberately a CLI transport,
not an OpenAI-compatible provider: authentication is owned by ``codex login``.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
from collections.abc import AsyncGenerator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from google.adk.models._capabilities import LlmCapabilities
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from pydantic import BaseModel

from fedotmas.common.logging import get_logger

_log = get_logger("fedotmas.codex_cli")
_HOST_PREFIXES = ("host/", "codex/")
_DEFAULT_TIMEOUT_SECONDS = 300.0


@dataclass(frozen=True)
class CodexCliResult:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


def is_codex_model(model: str | None) -> bool:
    return bool(model and model.startswith(_HOST_PREFIXES))


def native_codex_model(model: str) -> str:
    if not is_codex_model(model):
        raise ValueError(f"Not a host-native Codex model: {model}")
    return model.split("/", 1)[1]


def find_codex_cli() -> str | None:
    configured = os.getenv("CODEX_CLI_PATH", "").strip()
    if configured and Path(configured).is_file():
        return configured
    return shutil.which("codex") or shutil.which("codex.exe")


async def codex_login_status() -> tuple[bool, str]:
    executable = find_codex_cli()
    if not executable:
        return False, "Codex CLI не найден в PATH"
    process = await asyncio.create_subprocess_exec(
        executable,
        "login",
        "status",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        **_creation_options(),
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=10)
    except TimeoutError:
        process.kill()
        await process.wait()
        return False, "Проверка авторизации Codex CLI превысила 10 секунд"
    note = _decode(stdout or stderr).strip()
    return process.returncode == 0, note


async def run_codex_cli(
    model: str,
    prompt: str,
    *,
    output_schema: dict[str, Any] | None = None,
    timeout: float | None = None,
    workdir: str | Path | None = None,
) -> CodexCliResult:
    """Run one ephemeral Codex turn and return its final assistant message."""
    executable = find_codex_cli()
    if not executable:
        raise RuntimeError(
            "Codex CLI не найден. Установите Codex CLI и выполните `codex login`."
        )
    authenticated, auth_note = await codex_login_status()
    if not authenticated:
        raise RuntimeError(
            "Codex CLI не авторизован в подписке. Выполните `codex login` в "
            f"терминале и повторите запуск. Статус: {auth_note or 'нет сессии'}"
        )

    native_model = native_codex_model(model)
    root = Path(workdir or os.getenv("FEDOTMAS_CODEX_WORKDIR") or Path.cwd())
    root = root.resolve()
    if not root.is_dir():
        raise RuntimeError(f"Codex workdir does not exist: {root}")

    with tempfile.TemporaryDirectory(prefix="fedotmas-codex-") as temp_dir:
        final_path = Path(temp_dir) / "final.txt"
        command = [
            executable,
            "-a",
            "never",
            "exec",
            "-m",
            native_model,
            "--ignore-user-config",
            "--ignore-rules",
            "--ephemeral",
            "--sandbox",
            "read-only",
            "-C",
            str(root),
            "--color",
            "never",
            "--json",
            "--output-last-message",
            str(final_path),
        ]
        if output_schema is not None:
            schema_path = Path(temp_dir) / "schema.json"
            schema_path.write_text(
                json.dumps(output_schema, ensure_ascii=False), encoding="utf-8"
            )
            command.extend(["--output-schema", str(schema_path)])
        command.append("-")

        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            **_creation_options(),
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(prompt.encode("utf-8")),
                timeout=timeout or _DEFAULT_TIMEOUT_SECONDS,
            )
        except TimeoutError:
            process.kill()
            await process.wait()
            raise TimeoutError(f"Codex CLI model {native_model} timed out") from None

        stdout_text = _decode(stdout)
        stderr_text = _decode(stderr)
        if process.returncode != 0:
            diagnostic = _clean_diagnostic(stderr_text or stdout_text)
            if (
                "not logged in" in diagnostic.lower()
                or "unauthorized" in diagnostic.lower()
            ):
                diagnostic = (
                    "Codex CLI не авторизован в подписке. Выполните `codex login` "
                    "в терминале и повторите запуск. " + diagnostic
                )
            raise RuntimeError(
                f"Codex CLI ({native_model}) завершился с кодом "
                f"{process.returncode}: {diagnostic}"
            )

        text = (
            final_path.read_text(encoding="utf-8").strip()
            if final_path.is_file()
            else ""
        )
        if not text:
            raise RuntimeError(
                f"Codex CLI ({native_model}) не вернул финальное сообщение"
            )
        prompt_tokens, completion_tokens = _usage_from_jsonl(stdout_text)
        return CodexCliResult(
            text=text,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )


class CodexCliLlm(BaseLlm):
    """ADK ``BaseLlm`` backed by ephemeral ``codex exec`` calls."""

    @property
    def capabilities(self) -> LlmCapabilities:
        return LlmCapabilities(output_schema_and_tools=True)

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        del stream  # Codex emits one complete ADK response per turn.
        tools = _function_declarations(llm_request)
        response_schema = _response_schema(llm_request)
        cli_schema = response_schema
        if tools:
            cli_schema = _tool_decision_schema([item["name"] for item in tools])

        prompt = _render_request(llm_request, tools, response_schema)
        result = await run_codex_cli(self.model, prompt, output_schema=cli_schema)
        parts: list[types.Part]
        if tools:
            parts = [_decision_part(result.text, {item["name"] for item in tools})]
        else:
            parts = [types.Part.from_text(text=result.text)]

        usage = types.GenerateContentResponseUsageMetadata(
            prompt_token_count=result.prompt_tokens or None,
            candidates_token_count=result.completion_tokens or None,
            total_token_count=(result.prompt_tokens + result.completion_tokens) or None,
        )
        yield LlmResponse(
            content=types.Content(role="model", parts=parts),
            partial=False,
            turn_complete=True,
            usage_metadata=usage,
            model_version=native_codex_model(self.model),
        )


def _creation_options() -> dict[str, int]:
    return {"creationflags": 0x08000000} if os.name == "nt" else {}


def _decode(value: bytes) -> str:
    return value.decode("utf-8", errors="replace")


def _clean_diagnostic(value: str) -> str:
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    return " | ".join(lines[-8:])[:2000] or "неизвестная ошибка"


def _usage_from_jsonl(value: str) -> tuple[int, int]:
    prompt = completion = 0
    for line in value.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        usage = event.get("usage") or (event.get("item") or {}).get("usage") or {}
        prompt = max(prompt, int(usage.get("input_tokens") or 0))
        completion = max(completion, int(usage.get("output_tokens") or 0))
    return prompt, completion


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(by_alias=True, exclude_none=True)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    return value


def _response_schema(request: LlmRequest) -> dict[str, Any] | None:
    schema = request.config.response_json_schema or request.config.response_schema
    if schema is None:
        return None
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        return schema.model_json_schema()
    if isinstance(schema, dict):
        return schema
    if isinstance(schema, BaseModel):
        return schema.model_dump(by_alias=True, exclude_none=True)
    return _jsonable(schema)


def _function_declarations(request: LlmRequest) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for tool in request.config.tools or []:
        for declaration in tool.function_declarations or []:
            schema = declaration.parameters_json_schema
            if schema is None and declaration.parameters is not None:
                schema = declaration.parameters.model_dump(
                    by_alias=True, exclude_none=True
                )
            found.append(
                {
                    "name": declaration.name,
                    "description": declaration.description or "",
                    "parameters": _jsonable(schema or {"type": "object"}),
                }
            )
    return found


def _render_system(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, types.Content):
        return "\n".join(part.text or "" for part in value.parts or []).strip()
    if isinstance(value, types.Part):
        return value.text or ""
    if isinstance(value, list):
        return "\n".join(_render_system(item) for item in value).strip()
    return str(value)


def _render_contents(contents: list[types.Content]) -> str:
    lines: list[str] = []
    for content in contents:
        lines.append(f"[{content.role or 'unknown'}]")
        for part in content.parts or []:
            if part.text:
                lines.append(part.text)
            elif part.function_call:
                lines.append(
                    "FUNCTION_CALL "
                    + str(part.function_call.name)
                    + " "
                    + json.dumps(part.function_call.args or {}, ensure_ascii=False)
                )
            elif part.function_response:
                lines.append(
                    "FUNCTION_RESPONSE "
                    + str(part.function_response.name)
                    + " "
                    + json.dumps(
                        _jsonable(part.function_response.response or {}),
                        ensure_ascii=False,
                    )
                )
    return "\n".join(lines)


def _render_request(
    request: LlmRequest,
    tools: list[dict[str, Any]],
    response_schema: dict[str, Any] | None,
) -> str:
    blocks = [
        (
            "You are the language-model backend for one FEDOT.MAS/ADK agent turn. "
            "Follow the system instruction and conversation. Do not inspect files and "
            "do not use Codex shell tools. Return only the requested final response."
        ),
        "SYSTEM INSTRUCTION:\n" + _render_system(request.config.system_instruction),
        "CONVERSATION:\n" + _render_contents(request.contents),
    ]
    if tools:
        blocks.append(
            "AVAILABLE FUNCTIONS:\n"
            + json.dumps(tools, ensure_ascii=False, indent=2)
            + "\nChoose exactly one action. To call a function, set kind to "
            '"function_call", put its exact name in function_name, and encode '
            "the arguments object as JSON in arguments_json. To answer the user, "
            'set kind to "text". All four output fields are required; use an empty '
            'string and "{}" for unused fields.'
        )
    elif response_schema:
        blocks.append(
            "Return JSON that exactly matches this schema:\n"
            + json.dumps(response_schema, ensure_ascii=False, indent=2)
        )
    return "\n\n".join(blocks)


def _tool_decision_schema(names: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "enum": ["text", "function_call"]},
            "text": {"type": "string"},
            "function_name": {"type": "string", "enum": [""] + names},
            "arguments_json": {"type": "string"},
        },
        "required": ["kind", "text", "function_name", "arguments_json"],
        "additionalProperties": False,
    }


def _decision_part(value: str, allowed_names: set[str]) -> types.Part:
    try:
        decision = json.loads(value)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Codex tool decision is not JSON: {exc}") from exc
    if decision.get("kind") == "text":
        text = str(decision.get("text") or "").strip()
        if not text:
            raise RuntimeError("Codex returned an empty text decision")
        return types.Part.from_text(text=text)
    if decision.get("kind") != "function_call":
        raise RuntimeError(f"Unknown Codex decision kind: {decision.get('kind')!r}")
    name = str(decision.get("function_name") or "")
    if name not in allowed_names:
        raise RuntimeError(f"Codex requested unavailable function: {name!r}")
    try:
        arguments = json.loads(str(decision.get("arguments_json") or "{}"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Codex function arguments are not JSON: {exc}") from exc
    if not isinstance(arguments, dict):
        raise TypeError("Codex function arguments must be a JSON object")
    return types.Part.from_function_call(name=name, args=arguments)
