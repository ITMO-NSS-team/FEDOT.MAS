"""The live GUI defaults to an OpenRouter model that is present in its catalog."""

from __future__ import annotations

import ast
from pathlib import Path


def test_default_models_use_openrouter_catalog():
    source = (Path(__file__).resolve().parents[1] / "server/config.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    defaults = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "DEFAULT_OPENROUTER_MODEL":
                    defaults[target.id] = node.value.value

    catalog_names = {"_CODEX_MODELS", "_OPEN_SOURCE_OPENROUTER", "MODELS"}
    catalog_nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id in catalog_names
            for target in node.targets
        )
    ]
    catalog = {}
    exec(compile(ast.Module(body=catalog_nodes, type_ignores=[]), "catalog", "exec"), catalog)

    model = defaults["DEFAULT_OPENROUTER_MODEL"]
    assert model.startswith("openrouter/")
    assert model in {item["id"] for item in catalog["MODELS"]}
    assert 'os.getenv("GUI_MODEL", DEFAULT_OPENROUTER_MODEL)' in source
    assert 'os.getenv("GUI_JUDGE_MODEL", DEFAULT_OPENROUTER_MODEL)' in source
