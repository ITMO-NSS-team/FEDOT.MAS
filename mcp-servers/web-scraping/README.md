## web-scraping

FastMCP proxy for [Lightpanda](https://lightpanda.io/), read-only web scraping: extract markdown, links, structured data, and run JavaScript.

### Prerequisites

```bash
just lightpanda-install
# or manually:
curl -fsSL https://pkg.lightpanda.io/install.sh | bash
```

Verify: `just lightpanda-check`

### How it works

A thin FastMCP proxy wrapping `lightpanda mcp` via `StdioTransport`. All tools from the underlying server are proxied without modification. Can be served over stdio (local) or HTTP/SSE (hosted).

### Proxied tools

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
maw = MAW(mcp_servers=["web-scraping"])
```

### TLS

Lightpanda is launched with `--insecure_disable_tls_host_verification` because its built-in TLS stack fails certificate verification for most HTTPS sites (`PeerFailedVerification`). This is standard practice for headless browsers used in automation/scraping contexts.

### Known issues

- **`invalid body: json decoder` on shutdown** — When stdin closes (pipe EOF), Lightpanda writes a raw error string to stdout instead of a JSON-RPC error response. The MCP Python SDK handles this gracefully (logs a warning). Does not affect normal operation.
- **Telemetry 400 warnings** — Lightpanda sends telemetry to its server and logs `$scope=telemetry $level=warn $msg="server error" status=400` to stderr. These are harmless and don't affect operation. `--log_filter_scopes telemetry` exists in `--help` but only works in debug builds.
