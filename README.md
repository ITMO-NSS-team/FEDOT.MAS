<div align="center">

# `FEDOT.MAS`

**Multi-Agent Systems Generation**

</div>

FEDOT.MAS automatically generates and executes multi-agent pipelines from a plain-text task description.

## Repository map

| Package | Path | Description |
|---------|------|-------------|
| **fedot-mas** | [`packages/fedotmas`](packages/fedotmas) | Core library: meta-agent, pipeline builder & runner |
| **fedotmas-synapse** | [`packages/fedotmas-synapse`](packages/fedotmas-synapse) | Synapse Platform integration plugin |
| **mcp-servers** | [`mcp-servers/`](mcp-servers/) | Internal MCP-servers registry |


## Development

Install dev dependencies (ruff, ty, prek):

```
uv sync --group dev
```

### Pre-commit hooks

Install the hooks into your local repo:

```
prek install
```

After that, `ruff` and `ty` will run automatically before each commit.

### Running checks manually

```
uv run ruff check . --fix
uv run ruff format .
uv run ty check packages/
```
