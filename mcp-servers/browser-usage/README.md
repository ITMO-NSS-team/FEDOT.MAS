## browser-usage

FastMCP proxy for [browser-use](https://github.com/browser-use/browser-use), interactive browser automation with clicks, typing, screenshots, and content extraction.

### Prerequisites

Chromium-based browser installed. browser-use uses Playwright under the hood:

```bash
uvx playwright install chromium
```

### How it works

A thin FastMCP proxy wrapping `browser-use --mcp` via `StdioTransport`. All tools from the underlying server are proxied without modification. Can be served over stdio (local) or HTTP/SSE (hosted).

### Proxied tools

| Tool | Description |
|------|-------------|
| `browser_navigate` | Navigate to a URL |
| `browser_go_back` | Go back to previous page |
| `browser_click` | Click on an element |
| `browser_input` | Type text into an input field |
| `browser_select_option` | Select a dropdown option |
| `browser_scroll_down` | Scroll down the page |
| `browser_scroll_up` | Scroll up the page |
| `browser_scroll_to_text` | Scroll to specific text |
| `browser_screenshot` | Take a screenshot |
| `browser_get_dropdown_options` | Get dropdown options |
| `browser_tab_focus` | Focus a browser tab |
| `browser_tab_new` | Open a new tab |
| `browser_tab_close` | Close a tab |
| `browser_get_text` | Extract page text |
| `browser_read_links` | Get all links on page |
| `browser_switch_tab` | Switch between tabs |

### Usage

```python
maw = MAW(mcp_servers=["browser-usage"])
```

### Environment variables

| Variable | Description |
|----------|-------------|
| `BROWSER_USE_HEADLESS` | Run headless (default: `true`) |
| `BROWSER_USE_LLM_MODEL` | LLM model for extract/agent tools (default: `openai/gpt-4o-mini`) |
| `BROWSER_USE_ALLOWED_DOMAINS` | Comma-separated domain whitelist |
| `OPENAI_API_KEY` | Required by browser-use for LLM calls |
| `OPENAI_BASE_URL` | Custom OpenAI-compatible endpoint (e.g. OpenRouter) |

### Internal LLM usage

browser-use uses an LLM internally in two tools:

- **`browser_extract`** — structured data extraction via `ChatOpenAI` (default model: `gpt-4o-mini`)
- **`retry_with_browser_use_agent`** — spawns a full autonomous browser agent as a fallback (default model: `gpt-4o`)

Both respect `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and `BROWSER_USE_LLM_MODEL`.

**Token usage from these internal LLM calls is not visible** through the MCP protocol. The proxy returns tool results as text, but token counts from browser-use's internal agents are not propagated to the caller. Only the outer agent's tokens (prompt/completion for MCP tool calls) are tracked by fedotmas. To monitor internal usage, check your LLM provider dashboard (e.g. OpenRouter usage page).
