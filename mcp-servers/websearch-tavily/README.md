# websearch-tavily MCP server

Search via Tavily's Search API. The server exposes one search tool and keeps
configured API keys inside the process.

## Configuration

Set either a comma-separated key list or the existing single-key variable:

    TAVILY_API_KEYS=key1,key2,key3
    TAVILY_ROTATE_EVERY=50
    # Or:
    TAVILY_API_KEY=your-key

Whitespace around list entries is ignored, as are empty entries. When both
forms contain keys, TAVILY_API_KEYS takes priority. The default rotation
interval is 50 actual Tavily HTTP request attempts.

GAIA exposes one unprefixed `search` tool. It tries Tavily first and uses
SearXNG (`SEARXNG_URL`, default `http://localhost:18888`) only when Tavily is
unavailable. `FEDOTMAS_GAIA_MCP_SERVERS` still overrides the other server
choices; the separate SearXNG search tool is removed from this worker surface.

## Behavior

Each API request attempt consumes one rotation slot, including an attempt that
fails. HTTP 401, 402, 403, 429, 432, and 433 mark that key unavailable until
this server process restarts. The current search retries each remaining
usable key once, then falls back to SearXNG after Tavily keys are exhausted.
Transport errors and Tavily 5xx responses also use SearXNG; other non-key 4xx
responses are returned as request errors. A successful search with no results
returns an empty results list and no error.

The telemetry tool reports aggregate counts and safe labels such as
key_0. It never includes API key values.
