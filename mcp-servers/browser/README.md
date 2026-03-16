## browser

Headless browser MCP server powered by [Lightpanda](https://lightpanda.io/), a headless browser with built-in MCP support.

### Install

```bash
just lightpanda-install
# or manually:
curl -fsSL https://pkg.lightpanda.io/install.sh | bash
```

Verify: `just lightpanda-check`

### How it works

Lightpanda runs as `lightpanda mcp` over stdio. No Python server needed — the MCP protocol is built into the binary. Discovery picks it up via `mcp.command` in `pyproject.toml`.

### Tools

| Tool | Description |
|------|-------------|
| `goto` | Navigate to a URL |
| `markdown` | Extract page content as markdown |
| `semantic_tree` | Get page DOM as a semantic tree |
| `links` | List all links on the page |
| `interactiveElements` | List interactive elements (buttons, inputs, etc.) |
| `structuredData` | Extract JSON-LD / microdata / meta tags |
| `evaluate` | Run JavaScript on the page |

### Usage

```python
maw = MAW(mcp_servers=["browser"])
```

See `examples/tools/browser.py` for a complete example.
