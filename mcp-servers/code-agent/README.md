# code-agent MCP

The server exposes one high-level tool, `solve_with_code`, for bounded Python
computation over explicitly supplied files. It uses an internal model loop to
write and repair small scripts, then returns a compact answer, evidence, status,
files used, steps, errors, and nested token usage. It does not return scripts or
full execution history.

Python runs in a fresh E2B code-interpreter sandbox with outbound internet
disabled. Only caller-supplied files are staged. Input is limited to 10 files,
25 MB per file, and 50 MB total. The sandbox is killed after each call.
Available artifact types are CSV, TSV, XLSX, JSON, JSONL, TXT, Markdown, ZIP,
and PDF. The server does not replace the `document` MCP; retrieval-heavy tasks
are routed there.

## Configuration

GAIA worker model and provider credentials are inherited by default. Optional
code-agent-specific overrides are:

    CODE_AGENT_MODEL=...
    CODE_AGENT_API_KEY=...
    CODE_AGENT_BASE_URL=...

When a custom base URL is used, it must have its own matching API key. A model
override alone inherits the currently resolved GAIA worker provider. Code
execution requires:

    E2B_API_KEY=...

Default bounds are 5 Python executions, 60 seconds total, and 4,000 characters
of execution output. The tool allows up to 8 executions, 120 seconds, and 12,000
characters. It also limits final answers and evidence. Model usage and reported
cost are returned separately from GAIA worker usage.

## Example request

    solve_with_code(
        task="Sum the amounts for category supplies",
        files=["expenses.xlsx"],
        context="Use the current quarter sheet",
        max_steps=5
    )

Example response:

    {
      "status": "completed",
      "answer": "$80",
      "evidence": ["Sheet 'Expenses', row 12: category supplies, amount 80"],
      "files_used": ["expenses.xlsx"],
      "steps_taken": 3,
      "errors": [],
      "usage": {
        "llm_invocations": 4,
        "prompt_tokens": 900,
        "completion_tokens": 120,
        "total_tokens": 1020,
        "cost_usd": null,
        "available": true
      }
    }
