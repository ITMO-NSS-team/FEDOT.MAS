"""Check the model picker without starting infrastructure or loading keys."""

import ast
import re
from pathlib import Path


def test_model_catalog_matches_offline_demo():
    gui = Path(__file__).resolve().parents[1]
    tree = ast.parse((gui / "server/config.py").read_text(encoding="utf-8"))
    names = {"_CODEX_MODELS", "_OPEN_SOURCE_OPENROUTER", "MODELS"}
    assignments = [node for node in tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id in names for t in node.targets)]
    scope = {}
    exec(compile(ast.Module(body=assignments, type_ignores=[]), "catalog", "exec"), scope)
    models = scope["MODELS"]
    ids = [model["id"] for model in models]
    assert len(ids) == len(set(ids)) == 13
    assert all(m["open"] and m["id"].startswith("openrouter/") for m in models[:10])
    assert all(not m["open"] and m["id"].startswith("host/") for m in models[10:])
    demo = (gui / "tools/mock_backend.js").read_text(encoding="utf-8")
    demo_catalog = demo.split("const MODELS = [", 1)[1].split("];", 1)[0]
    assert re.findall(r'id: "([^"]+)"', demo_catalog) == ids
