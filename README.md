# fedot-mas

## Setup

**Prerequisites**: Python 3.12+, [uv](https://docs.astral.sh/uv/)

1. Clone and install

```
uv sync
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
- **ty** type checking (`ty check src/`)

Install the hooks into your local repo:

```
prek install
```

After that, `ruff` and `ty` will run automatically before each commit.

### Running checks manually

```
uv run ruff check src/ --fix   # lint + auto-fix
uv run ruff format src/        # format
uv run ty check src/            # type check
```
