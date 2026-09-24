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
maw = MAW(mcp_servers=["web-scraping"])
```

### Extract fallback

`extract` runs the caller's schema against the page inside lightpanda, and fails in three different ways that all mean "the page is fine, the schema was not":

| Cause | Error |
|---|---|
| No top-level key matched anything | `extract: no schema selector matched any element` (by design) |
| A non-string value used as a selector spec — e.g. `"limit": 1` at the top level instead of inside an array spec | `Cannot read properties of null (reading 'trim')` |
| A selector CSS does not support, e.g. `p:contains('...')` | `The string did not match the expected pattern` |

Nothing in those tells the caller whether the page lacks the data or the schema was wrong, so agents tend to give up on the page and go back to searching — in one GAIA run the first such failure was followed by six more web searches for values that were on pages the agent had already loaded.

`ExtractMarkdownFallback` appends the page, as markdown, to the failed result, prefixed with the original error and a note to read the values out of it. `markdown` is tried against the current page first, then against the last URL `goto` navigated to. If markdown fails or comes back blank, the original `extract` error is returned untouched.

The result stays an error. Flipping it to success would hand markdown's content to the client's validation of `extract`'s own output schema (`mcp.client.session` validates only non-error results), and a caller that asked for fields should not receive a whole page as if its schema had matched. A successful `extract` is passed through untouched.

### TLS

Lightpanda is launched with `--insecure_disable_tls_host_verification` because its built-in TLS stack fails certificate verification for most HTTPS sites (`PeerFailedVerification`). This is standard practice for headless browsers used in automation/scraping contexts.

### Known issues

- **`invalid body: json decoder` on shutdown** — When stdin closes (pipe EOF), Lightpanda writes a raw error string to stdout instead of a JSON-RPC error response. The MCP Python SDK handles this gracefully (logs a warning). Does not affect normal operation.
- **Telemetry 400 warnings** — Lightpanda sends telemetry to its server and logs `$scope=telemetry $level=warn $msg="server error" status=400` to stderr. These are harmless and don't affect operation. `--log_filter_scopes telemetry` exists in `--help` but only works in debug builds.
- **`CouldntResolveHost`** — Navigation is performed by the proxied Lightpanda process, so this error points to DNS or outbound network access where that process runs. Check resolution for the failing host and an unrelated public host from the same environment; if both fail, fix the environment's resolver/network access rather than adding a site-specific fallback. In the current investigation, `apod.nasa.gov`, `nasa.gov`, and `example.com` all failed host lookup, indicating an environment-level issue.
