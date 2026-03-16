# websearch-searxng MCP server

Web search via self-hosted SearXNG. Registered automatically in FEDOT.MAS.

## Setup

```bash
just searxng-install   # clone searxng-docker, configure JSON output
just searxng-start     # docker compose up
just searxng-stop      # docker compose down
just searxng-status    # docker compose ps
```

Installs to platform data directory (`~/.local/share/fedotmas/searxng` on Linux, `~/Library/Application Support/fedotmas/searxng` on macOS). Override with `SEARXNG_DIR` env var. Default port: `18888`. Override via `SEARXNG_URL` in `.env`.

## Usage

```python
AgentConfig(
    name="researcher",
    instruction="Search the web and answer: {user_query}",
    output_key="result",
    tools=["websearch-searxng"],
)
```
