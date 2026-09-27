"""Generate a portable source bundle for a GUI MAS/MAW without running it."""

from __future__ import annotations

import io
import json
import re
import zipfile

from fedotmas import MASConfig, MAWConfig

from .agent_names import latinize_mas
from .normalize import _builtin_names, _valid_name, sanitize_config
from .schemas import CodeExportIn


# Prompts are user data, so redact recognizable credentials even when someone
# pasted one into an instruction. Arbitrary private data still needs human review.
_CREDENTIAL = re.compile(
    r"sk-(?:or-v1-)?[A-Za-z0-9_-]{16,}|"
    r"(?i:bearer)\s+[^\s,;]+|"
    r"(?i:(?:api[_-]?key|authorization|password|token))\s*[:=]\s*[^\s,;]+"
)


def _redact(value):
    if isinstance(value, dict):
        return {key: _redact(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, str):
        return _CREDENTIAL.sub("[REDACTED]", value)
    return value


def _custom_names(custom, tools: list[str]) -> list[str]:
    """Match GUI name normalization without embedding MCP URLs or headers."""
    reserved = _builtin_names()
    used = set(tools)
    names = []
    for item in custom or []:
        name = _valid_name(item.name)
        if name in used or name in reserved:
            name = _valid_name(f"custom_{name}")
            while name in used:
                name += "_"
        used.add(name)
        names.append(name)
    return list(dict.fromkeys(names))


RUNNER = '''"""Run the exported FEDOT.MAS system. Pass the task on the command line."""
import asyncio
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

from fedotmas import MAS, MAW, MASConfig, MAWConfig
from fedotmas.mcp import HttpMCPServer, StdioMCPServer, resolve_mcp_registry
from fedotmas.plugins import LoggingPlugin, UnknownToolRecoveryPlugin


ROOT = Path(__file__).resolve().parent


async def main():
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    config_data = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    is_mas = manifest["kind"] == "mas"
    config = (MASConfig if is_mas else MAWConfig).model_validate(config_data)
    servers = dict(resolve_mcp_registry(manifest["tools"]) or {})

    # Match the GUI command for this built-in MCP server when installed from source.
    rubber = servers.get("rubber-recipe-predictor")
    if isinstance(rubber, StdioMCPServer) and "--directory" in rubber.args:
        directory = rubber.args[rubber.args.index("--directory") + 1]
        servers["rubber-recipe-predictor"] = replace(
            rubber, args=("run", "--directory", directory,
                          "python", "-m", "mcp_rubber_recipe_predictor.server"))

    custom = manifest["custom_mcp"]
    if custom:
        # URLs and credentials must be supplied separately: they are not in the ZIP.
        supplied = json.loads(os.environ.get("FEDOTMAS_CUSTOM_MCP", "{}"))
        for name in custom:
            settings = supplied.get(name)
            if not isinstance(settings, dict) or not settings.get("url"):
                raise ValueError(f"Set FEDOTMAS_CUSTOM_MCP URL for {name}")
            servers[name] = HttpMCPServer(url=settings["url"],
                                          headers=settings.get("headers") or {},
                                          description=f"Custom MCP server: {name}",
                                          tags=("custom",))

    model = manifest["model"]
    system = (MAS if is_mas else MAW)(
        worker_models=[model], mcp_servers=servers,
        plugins=[LoggingPlugin(), UnknownToolRecoveryPlugin()])
    query = " ".join(sys.argv[1:]).strip() or input("Задача для МАС: ").strip()
    if not query:
        raise ValueError("Enter a non-empty task")
    result = await system.build_and_run(config, query)
    state = result if isinstance(result, dict) else getattr(result, "state", {})
    print(json.dumps(dict(state), ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
'''


def build_code_archive(body: CodeExportIn, *, tools: list[str], default_model: str) -> bytes:
    if body.kind not in {"mas", "maw"}:
        raise ValueError("Допустимый тип системы: mas или maw")

    config = (MASConfig if body.kind == "mas" else MAWConfig).model_validate(body.config)
    custom = _custom_names(body.custom_mcp, tools)
    # The GUI performs these transformations immediately before build_and_run.
    # Export the runnable form, with original prompts/topology and active model.
    config = sanitize_config(config, body.kind, custom,
                             available_tools=set(tools) | set(custom))
    if body.kind == "mas":
        latinize_mas(config)
    if body.model:
        agents = (getattr(config, "agents", None)
                  or [config.coordinator, *config.workers])
        for agent in agents:
            agent.model = body.model

    manifest = {"kind": body.kind, "tools": tools, "custom_mcp": custom,
                "model": body.model or default_model}
    readme = (
        "# Исходный код выбранной МАС\n\n"
        "`config.json` содержит агентов, их инструкции и схему взаимодействия; "
        "`run.py` создаёт и запускает систему на библиотеке FEDOT.MAS. "
        "`manifest.json` фиксирует тип, модель и выбранные инструменты. "
        "Экспорт не запускает агентов и не обращается к LLM.\n\n"
        "## Запуск\n\n"
        "Установите Python 3.12+ и FEDOT.MAS из исходного репозитория "
        "(вместе с нужными MCP-серверами из `mcp-servers/`), настройте "
        "`OPENROUTER_API_KEY` или ключ выбранного провайдера, затем выполните "
        "`python run.py \"Ваша задача\"`. Модель, работающая через Codex CLI "
        "(`host/…`), требует установленного и авторизованного CLI. "
        "Инструментам могут понадобиться отдельные сервисы и зависимости.\n\n"
        "URL и заголовки пользовательских MCP-серверов не экспортируются. "
        "Если они используются, перед запуском задайте переменную окружения "
        "`FEDOTMAS_CUSTOM_MCP` как JSON-объект: "
        "`{\"имя_сервера\": {\"url\": \"https://…\", "
        "\"headers\": {\"Authorization\": \"Bearer …\"}}}`. "
        "Имена нужных серверов перечислены в `manifest.json`.\n\n"
        "Загруженные в GUI файлы, история выполнения, ключи и исходный текст "
        "запроса в архив не включаются. Проверяйте инструкции агентов перед "
        "передачей архива другим людям: в них могут быть частные данные.\n"
    )
    contents = {
        "run.py": RUNNER,
        "config.json": json.dumps(_redact(config.model_dump(mode="json")), ensure_ascii=False, indent=2),
        "manifest.json": json.dumps(_redact(manifest), ensure_ascii=False, indent=2),
        "README.md": readme,
    }
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in contents.items():
            archive.writestr(name, content.encode("utf-8"))
    return output.getvalue()
