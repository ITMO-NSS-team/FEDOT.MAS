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

## Quick start

### With [just](https://github.com/casey/just)

Create a virtual environment and install both packages:

```
just venv
```

Set up a full dev environment (linters, hooks):

```
just venv-dev
```

### With [uv](https://github.com/astral-sh/uv)

Create a virtual environment and install workspace packages:

```sh
uv sync
cp -n .env.example .env 2>/dev/null || true
```

For development (linters, type checker, pre-commit hooks):

```sh
uv sync --group dev
uv run prek install
```

## Development

### Linting & type checking

```
just lint        # ruff check + format
just typecheck   # ty check
just check       # both
```

Or manually:

```sh
uv run ruff check --fix .
uv run ruff format .
uv run ty check
```

### Tests

```
just test-unit
```

Or manually:

```sh
uv run pytest
```
