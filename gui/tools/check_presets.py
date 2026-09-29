"""Проверка сценариев presets.js по правилам моделей fedotmas.

Запуск:  python3 gui/tools/check_presets.py
Нужен node (используется только для чтения presets.js как объекта JS).
Проверяются: имена и output_key агентов, ссылки pipeline на агентов, ссылки на
переменные состояния {var}, префикс провайдера у моделей, существование MCP-серверов,
согласованность журнала запуска с конфигурацией и с подписью времени/токенов.
"""
import json, re, sys, pathlib, subprocess

HERE = pathlib.Path(__file__).resolve().parent
STATIC = HERE.parent / "static"   # presets.js лежит рядом с интерфейсом
# Кейс-пресеты (*_preset.js) регистрируются в window.STARTUP_PRESETS —
# загружаем их вслед за presets.js и проверяем теми же правилами.
_case_files = sorted(p.name for p in STATIC.glob("*_preset.js"))
_loads = "".join(f"require('./{name}');" for name in ["presets.js", *_case_files])
try:
    dump = subprocess.run(
        ["node", "-e",
         f"global.window={{}};{_loads}"
         "console.log(JSON.stringify({presets:window.PRESETS,"
         "startup:window.STARTUP_PRESETS||[],"
         "mcp:window.MCP_SERVERS,meta:window.META_MODEL}))"],
        cwd=STATIC, capture_output=True, text=True, check=True).stdout
except (OSError, subprocess.CalledProcessError) as exc:
    sys.exit(f"не удалось прочитать пресеты через node: {exc}")
data = json.loads(dump)
presets, MCP = data["presets"] + data["startup"], data["mcp"]
REAL_TOOLS = {"browser-usage","document","download","media","sandbox","sandbox-light",
              "sequential-thinking","web-scraping","websearch-searxng","youtube-transcript"}
# Кейс-серверы из mcp-servers/ репозитория тоже реальны — читаем их имена из
# [tool.fedotmas.mcp] в pyproject.toml, чтобы список не отставал от репо.
import tomllib
for _pp in (HERE.parent.parent / "mcp-servers").glob("*/pyproject.toml"):
    try:
        _name = tomllib.loads(_pp.read_text(encoding="utf-8"))["tool"]["fedotmas"]["mcp"]["name"]
        REAL_TOOLS.add(_name)
    except (KeyError, OSError, tomllib.TOMLDecodeError):
        pass
errs, warns = [], []

def err(p, m): errs.append(f"[{p}] {m}")
def warn(p, m): warns.append(f"[{p}] {m}")

def check_model(p, name, model):
    if model is not None and "/" not in model:
        err(p, f"{name}: модель '{model}' без префикса провайдера (validate_model_name)")

def refs(node, acc):
    if node.get("type") == "agent":
        if node.get("agent_name"): acc.add(node["agent_name"])
    for c in node.get("children", []): refs(c, acc)

def validate_node(p, node, names):
    t = node.get("type")
    if t == "agent":
        if not node.get("agent_name"): err(p, "узел agent без agent_name")
        elif node["agent_name"] not in names: err(p, f"pipeline ссылается на неизвестного агента '{node['agent_name']}'")
        if node.get("children"): err(p, f"узел agent '{node.get('agent_name')}' имеет children")
    else:
        if t not in ("sequential", "parallel", "loop"): err(p, f"неизвестный тип узла '{t}'")
        if not node.get("children"): err(p, f"узел '{t}' без children")
        if node.get("agent_name"): err(p, f"узел '{t}' имеет agent_name")
        if t != "loop" and node.get("max_iterations") is not None: err(p, f"max_iterations у узла '{t}'")
        for c in node.get("children", []): validate_node(p, c, names)

def terminal(node):
    if node.get("type") in ("agent", "parallel"): return node
    ch = node.get("children") or []
    return terminal(ch[-1]) if ch else node

for pr in presets:
    p, cfg, kind = pr["id"], pr["config"], pr["kind"]
    agents = [cfg["coordinator"], *cfg["workers"]] if kind == "mas" else cfg["agents"]
    names = [a["name"] for a in agents]

    # общее
    if len(set(names)) != len(names): err(p, f"дублирующиеся имена агентов: {names}")
    for a in agents:
        check_model(p, a["name"], a.get("model"))
        for t in a.get("tools", []):
            if t not in REAL_TOOLS: err(p, f"{a['name']}: неизвестный MCP-сервер '{t}'")
            if t not in MCP: err(p, f"{a['name']}: у инструмента '{t}' нет описания в MCP_SERVERS")
        if not a.get("instruction"): err(p, f"{a['name']}: пустая instruction")

    if kind == "mas":
        if len(cfg["workers"]) < 1: err(p, "нужен хотя бы один worker")
        for a in agents:
            if not a.get("description"): err(p, f"{a['name']}: пустой description (нужен AutoFlow для маршрутизации)")
        if cfg["coordinator"].get("output_key"): warn(p, "у координатора задан output_key")
    else:
        keys = [a["output_key"] for a in agents]
        if len(set(keys)) != len(keys): err(p, f"дублирующиеся output_key: {keys}")
        for a in agents:
            if not a.get("output_key"): err(p, f"{a['name']}: отсутствует output_key")
        validate_node(p, cfg["pipeline"], set(names))
        used = set(); refs(cfg["pipeline"], used)
        if set(names) - used: warn(p, f"агенты не задействованы в pipeline: {sorted(set(names)-used)}")
        if terminal(cfg["pipeline"]).get("type") == "parallel": warn(p, "pipeline заканчивается parallel-узлом")
        # ссылки на state-переменные вида {var}
        state = {"user_query"} | set(keys)
        for a in agents:
            for var in re.findall(r"\{(\w+)\}", a["instruction"]):
                if var not in state: err(p, f"{a['name']}: ссылка на неизвестную переменную состояния {{{var}}}")

    # согласованность журнала запуска
    for e in pr["trace"]:
        if e["agent"] not in names: err(p, f"trace: неизвестный агент '{e['agent']}'")
        for t in (e.get("tool"), e.get("tool2")):
            if t and t not in ("exit_loop", "transfer_to_agent"):   # встроенные инструменты ADK
                owner = next((a for a in agents if a["name"] == e["agent"]), {})
                if t not in owner.get("tools", []): err(p, f"trace: {e['agent']} вызывает '{t}', которого нет в его tools")
        for f in ("phase", "text", "ms"):
            if f not in e: err(p, f"trace: событие {e['agent']} без поля '{f}'")
    if not any(e.get("final") for e in pr["trace"]): warn(p, "в журнале нет финального события")

    # согласованность подписей
    tok = sum(e.get("tokens", 0) for e in pr["trace"])
    m = re.search(r"·\s*([\d,]+)к токенов", pr["auto"])
    if m:
        declared = float(m.group(1).replace(",", ".")) * 1000
        if abs(declared - tok) / max(tok, 1) > 0.05:
            err(p, f"подпись '{pr['auto']}' не сходится с суммой токенов журнала ({tok})")
    # без парсимого времени интерфейс считает часы по журналу (factor=1) — это допустимо
    if not re.search(r"\d+\s*(мин|с)", pr["auto"]): warn(p, f"из подписи '{pr['auto']}' не парсится время — часы пойдут по журналу")

print(f"проверено сценариев: {len(presets)}")
print(f"\nОШИБКИ ({len(errs)}):"); [print(" ✗", e) for e in errs] or print("  нет")
print(f"\nПРЕДУПРЕЖДЕНИЯ ({len(warns)}):"); [print(" !", w) for w in warns] or print("  нет")
sys.exit(1 if errs else 0)
