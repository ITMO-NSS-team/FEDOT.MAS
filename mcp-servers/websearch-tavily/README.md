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

GAIA keeps SearXNG enabled by default. It adds Tavily by default when a key is
configured. Set FEDOTMAS_GAIA_SEARCH_PROVIDERS to searxng, tavily, or
searxng,tavily to choose the providers while retaining the rest of GAIA's MCP
servers. The existing FEDOTMAS_GAIA_MCP_SERVERS setting still overrides the
full server list.

## Behavior

Each API request attempt consumes one rotation slot, including an attempt that
fails. A 401, 402, 403, or 429 response marks that key unavailable until this
server process restarts and retries on the next usable key. One search tries
each usable key at most once. Other provider failures return a structured
TAVILY_PROVIDER_ERROR; a successful search with no results returns an empty
results list and no error.

The tavily_telemetry tool reports aggregate counts and safe labels such as
key_0. It never includes API key values.
