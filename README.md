<div align="center">

# `FEDOT.MAS`

**Multi-Agent Systems Generation**

</div>

FEDOT.MAS automatically generates and executes multi-agent pipelines from a plain-text task description. A meta-agent analyses the task, designs a pipeline config (sequential, parallel, or loop), and builds an ADK agent tree that runs the plan.

## Monorepo structure

| Package | Path | Description |
|---------|------|-------------|
| **fedot-mas** | [`packages/fedotmas`](packages/fedotmas) | Core library: meta-agent, pipeline builder & runner |
| **fedotmas-synapse** | [`packages/fedotmas-synapse`](packages/fedotmas-synapse) | CodeSynapse integration plugin |

Additional workspace members live in `mcp-servers/`.

## Quick start

**Prerequisites**: Python 3.12+, [uv](https://docs.astral.sh/uv/)

1. Clone and install

```
uv sync --all-packages
source .venv/bin/activate
```

2. Configure environment

```
cp .env.example .env
```

Open `.env` and set your model provider.

Default model and agent settings live in `config.toml`. Environment variables override them (see `.env.example` for the full list).

3. Run the basic example

```
uv run python examples/basic_usage.py
```

## Development

Install dev dependencies (ruff, ty, prek):

```
uv sync --group dev
```

### Pre-commit hooks

The project uses [pre-commit](https://pre-commit.com/) hooks that run on every commit:

- **ruff** linting with auto-fix + formatting
- **ty** type checking

Install the hooks into your local repo:

```
prek install
```

After that, `ruff` and `ty` will run automatically before each commit.

### Running checks manually

```
uv run ruff check . --fix
uv run ruff format .
uv run ty check packages/fedotmas/src/
```
