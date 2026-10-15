# websearch-searxng MCP server

Web search via self-hosted SearXNG. Registered automatically in FEDOT.MAS.

## Setup

```bash
just searxng-install   # generate Docker Compose and SearXNG configuration
just searxng-start     # docker compose up
just searxng-stop      # docker compose down
just searxng-status    # docker compose ps
```

Installs to platform data directory (`~/.local/share/fedotmas/searxng` on Linux, `~/Library/Application Support/fedotmas/searxng` on macOS). Override with `SEARXNG_DIR` env var. Default port: `18888`.

To use an existing SearXNG instance, set `SEARXNG_URL` in `.env`.
This changes the MCP client's target URL, not the port of the local container.
To start the local container on a different port, use the same override when
installing and starting it (commands run from the repository root):

```bash
just searxng_port=18889 searxng-install
just searxng_port=18889 searxng-start
```

Then set `SEARXNG_URL=http://localhost:18889` in `.env`.

## Usage

```python
from fedotmas import MAW

maw = MAW(mcp_servers=["websearch-searxng"])
```
