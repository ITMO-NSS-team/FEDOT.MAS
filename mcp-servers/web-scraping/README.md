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

A thin FastMCP proxy wrapping `lightpanda mcp` via `StdioTransport`. Tools from the underlying server are proxied as they are, with one middleware on top (see *Extract fallback*). Can be served over stdio (local) or HTTP/SSE (hosted).

### Proxied tools

| Tool | Description |
|------|-------------|
| `goto` | Navigate to a URL |
| `markdown` | Extract page content as markdown |
| `semantic_tree` | Get page DOM as a semantic tree |
| `links` | List all links on the page |
| `interactiveElements` | List interactive elements (buttons, inputs, etc.) |
| `extract` | Pull fields off the page with a JSON schema of CSS selectors |
| `structuredData` | Extract JSON-LD / microdata / meta tags |
| `evaluate` | Run JavaScript on the page |

### Usage

```python
from fedotmas import MAW

maw = MAW(mcp_servers=["web-scraping"])
```

### Extract fallback

When `extract` returns a selector/schema error, the proxy appends page markdown
so the caller can read the content directly. It tries the current page first,
then the last URL passed to `goto`. If markdown is unavailable, the original
error is returned unchanged.

The fallback preserves the error status and does not produce the requested
structured fields. Successful `extract` results pass through unchanged.

### TLS

The proxy launches Lightpanda with `--insecure_disable_tls_host_verification`.
TLS host verification is disabled, so the connection does not verify the
server's identity.

### Troubleshooting

- **`CouldntResolveHost`** — check DNS resolution and outbound network access
  from the environment running Lightpanda.
- **`invalid body: json decoder` during shutdown** — Lightpanda may emit a
  non-JSON-RPC message when stdin closes.
- **Telemetry HTTP 400 warnings** — these concern Lightpanda's telemetry
  endpoint. Check the tool result separately to determine whether navigation
  or extraction succeeded.
