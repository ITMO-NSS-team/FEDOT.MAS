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

Open `.env` and set at your model provider.

Default model and agent settings live in `config.toml`. Environment variables override them (see `.env.example` for the full list).

3. Run the basic example

```
uv run python examples/basic_usage.py
```
