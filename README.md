<div align="center">

# `FEDOT.MAS`

**Multi-Agent Systems Generation**

</div>

FEDOT.MAS automatically generates and executes multi-agent pipelines from a plain-text task description.

## Quick start

Managed with [uv](https://github.com/astral-sh/uv).

**With [just](https://github.com/casey/just):**

Create a virtual environment and install both packages:

```sh
just venv
```

**Or manual:**

```sh
uv sync
cp -n .env.example .env 2>/dev/null || true
```

## Agent skill

The reusable [FEDOT.MAS subagents skill](skills/fedot-mas-subagents/SKILL.md)
helps agent hosts design and run bounded `MAW` workflows. It also includes a
file bridge for host-native models that do not expose a compatible API.

## Development

**With just:**

```sh
just venv-dev
```

**Or manual:**

```sh
uv sync --group dev
uv run prek install
```

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

```sh
just test-unit
```

### Docs

```sh
uv sync --group docs
zensical serve
```
