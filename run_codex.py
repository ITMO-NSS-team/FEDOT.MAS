import json
import subprocess
from pathlib import Path

FEDOT_RESULTS = Path("/Users/daniel/Documents/nss_lab/FEDOT.MAS/benchmarks/gaia/gaia_logs/run_d78541b5-ba6a-4dbe-9c8c-590457130ea9/results.json")   # поставь путь к своему results
REPO = Path("/Users/daniel/Documents/nss_lab/FEDOT.MAS")
GAIA_DATA = REPO / "benchmarks/gaia/gaia_data/2023/validation"
OUT = Path("codex_runs")

MODEL = "gpt-5.6-terra"
EFFORT = "medium"

TASK_IDS = [
    "9b54f9d9-35ee-4a14-b62f-d130ea00317f",
    "56db2318-640f-477a-a82f-bc93ad13e882",
    "e961a717-6b25-4175-8a68-874d28190ee4",
    "851e570a-e3de-4d84-bcfa-cc85578baa59",
    "0bdb7c40-671d-4ad1-9ce3-986b159c0ddc",
]

OUT.mkdir(exist_ok=True)

data = json.loads(FEDOT_RESULTS.read_text())

rows = {
    r["task_id"]: r
    for r in data["results"]
}

summary = []

for task_id in TASK_IDS:
    row = rows[task_id]
    question = row["question"]

    attachments = [
        p for p in GAIA_DATA.glob(f"{task_id}.*")
        if p.is_file()
    ]

    attachment_text = ""
    if attachments:
        attachment_text = "\n".join(
            f"Attached/local file: {p.resolve()}"
            for p in attachments
        )

    prompt = f"""
Solve the following GAIA benchmark task.

You have access to local files, shell/code tools, and web search.
Use them as needed. Inspect any provided local file directly.
Do not use or search for the benchmark ground truth.

{attachment_text}

Question:
{question}

Return the final answer clearly at the end.
""".strip()

    trace_path = OUT / f"{task_id}.jsonl"

    cmd = [
        "codex", "exec",
        "--json",
        "--ephemeral",
        "--model", "gpt-5.6-terra",
        "--config", 'model_reasoning_effort="medium"',
        "--config", 'web_search="live"',
        "--config", "sandbox_workspace_write.network_access=true",
        "--sandbox", "workspace-write",
        "--cd", str(REPO),
        prompt,
    ]

    print(f"\n=== {task_id} ===")

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    trace_path.write_text(proc.stdout)

    events = []
    for line in proc.stdout.splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            pass

    completed = [
        e for e in events
        if e.get("type") == "turn.completed"
    ]

    usage = completed[-1].get("usage", {}) if completed else {}

    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)

    cached_tokens = (
        usage.get("cached_input_tokens")
        or usage.get("input_tokens_details", {}).get("cached_tokens", 0)
        or 0
    )

    # Current GPT-5.6 Terra standard prices
    uncached_tokens = max(0, input_tokens - cached_tokens)

    token_cost = (
        uncached_tokens * 2.00 / 1_000_000
        + cached_tokens * 0.20 / 1_000_000
        + output_tokens * 12.00 / 1_000_000
    )

    # Count hosted web-search calls visible in the Codex trace.
    web_calls = sum(
        1
        for e in events
        if (
            e.get("item", {}).get("type") in
            {"web_search", "web_search_call"}
        )
    )

    web_cost = web_calls * 0.01

    summary.append({
        "task_id": task_id,
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "web_search_calls": web_calls,
        "token_cost_usd": token_cost,
        "web_cost_usd": web_cost,
        "estimated_cost_usd": token_cost + web_cost,
        "exit_code": proc.returncode,
    })

    if proc.stderr:
        (OUT / f"{task_id}.stderr").write_text(proc.stderr)

(OUT / "summary.json").write_text(
    json.dumps(summary, indent=2)
)

print("\n=== SUMMARY ===")

for x in summary:
    print(
        x["task_id"],
        f'tokens={x["total_tokens"]:,}',
        f'cost=${x["estimated_cost_usd"]:.4f}',
    )

print(
    "\nTOTAL TOKENS:",
    f'{sum(x["total_tokens"] for x in summary):,}'
)

print(
    "TOTAL COST:",
    f'${sum(x["estimated_cost_usd"] for x in summary):.4f}'
)