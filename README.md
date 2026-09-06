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

### MCP servers

The servers under `mcp-servers/` each carry their own environment, built on first
use. Build them up front, or the first agent run pays dependency resolution inside
its session-ready timeout and fails:

```sh
just mcp-sync
```

Some servers also need a browser or a Docker container:

```sh
just deps-external
```

`sandbox` also needs `E2B_API_KEY`, which is set in `.env` by hand.

Discovery lists a server whether or not its prerequisites are present, and so does
the server's own tool list — a missing one surfaces only mid-run. To see the real
state of every server beforehand:

```sh
just doctor
```

Install only what the task needs and pass those names to `mcp_servers=`.

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
