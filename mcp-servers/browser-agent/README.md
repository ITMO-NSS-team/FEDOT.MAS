# browser-agent MCP

`browser-agent` exposes one high-level `complete_browser_task` tool. Browser-Use runs its Chromium session and internal LLM loop inside this MCP, then returns a compact finding, visited URLs, status, and short errors. Screenshots and internal action histories are not returned.

Configure `OPENAI_API_KEY` or `OPENROUTER_API_KEY`. Optional settings:

| Variable | Default | Purpose |
| --- | --- | --- |
| `BROWSER_AGENT_MODEL` | `gpt-4o-mini` (OpenAI) or `openai/gpt-4o-mini` (OpenRouter) | Browser-Use model |
| `BROWSER_AGENT_BASE_URL` | `OPENAI_BASE_URL`, or OpenRouter's API URL when using its key | OpenAI-compatible endpoint |
| `BROWSER_AGENT_API_KEY` | `OPENROUTER_API_KEY` or `OPENAI_API_KEY` | Explicit LLM API key |

The browser runs headless and uses a fresh profile per task. Install the Chromium runtime with `just browser-use-install`; a missing LLM key or browser runtime is returned with `status="blocked"`.

Use `websearch-searxng` to discover sources, `web-scraping` to extract known pages directly, and `browser-agent` when the task needs interaction or multiple browser steps. Use `download` for files and `sandbox` for computation and file operations.
