# browser-agent MCP

`browser-agent` exposes one high-level `complete_browser_task` tool. Browser-Use runs its Chromium session and internal LLM loop inside this MCP, then returns compact findings, visited URLs, status, short errors, and nested LLM usage. Screenshots and internal action histories are not returned. Failed, incomplete, and blocked runs set the MCP error flag and a `BROWSER_AGENT_*` error code.

Configure `OPENAI_API_KEY` or `OPENROUTER_API_KEY`. Provider settings are selected in this order: `BROWSER_AGENT_*`, `FEDOTMAS_GAIA_WORKER_*`, OpenRouter, then OpenAI. A model-only override keeps the selected endpoint and key. A key-only Browser Agent override uses the OpenAI endpoint; a key-only GAIA worker override uses OpenRouter. Custom endpoints require a matching API key.

| Variable | Default | Purpose |
| --- | --- | --- |
| `BROWSER_AGENT_MODEL` | `gpt-4o-mini` (OpenAI) or `openai/gpt-4o-mini` (OpenRouter) | Browser-Use model |
| `BROWSER_AGENT_BASE_URL` | Selected provider endpoint | OpenAI-compatible endpoint |
| `BROWSER_AGENT_API_KEY` | Selected provider key | Explicit LLM API key |

The browser runs headless and uses a fresh profile per task. Install the Chromium runtime with `just browser-use-install`, which uses this MCP's pinned Browser-Use environment. A missing LLM key or browser runtime returns `status="blocked"`. GAIA limits `complete_browser_task` to three calls per agent by default; set `FEDOTMAS_GAIA_BROWSER_AGENT_LIMIT` to adjust it. GAIA diagnostics store Browser-Use tokens and LLM invocation counts separately from outer MAW tokens.

Use `websearch-searxng` to discover sources, `web-scraping` to extract known pages directly, and `browser-agent` when the task needs interaction or multiple browser steps. Use `download` for files and `sandbox` for computation and file operations.
