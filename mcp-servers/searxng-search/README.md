# searxng-search MCP server

Web search via self-hosted SearXNG. Registered automatically in FEDOT.MAS.

## Prerequisites

A running SearXNG instance with JSON output enabled — add to `settings.yml`:

```yaml
search:
  formats:
    - html
    - json
```

Set `SEARXNG_URL` in `.env` to point to your instance (default: `http://localhost:8888`).

## Usage in FEDOT.MAS

In auto mode the meta-agent assigns the tool automatically when the task requires web search:

```python
mas = MAS()
state = await mas.run("What are the latest AI research papers from 2025?")
```

To assign it explicitly to an agent:

```python
AgentConfig(
    name="researcher",
    instruction="Search the web and answer: {user_query}",
    output_key="result",
    tools=["searxng-search"],
)
```
