venv:
    uv sync
    cp -n .env.example .env 2>/dev/null || true

venv-dev:
    uv sync --group dev
    uv run prek install
    @echo "Dev environment ready"

bifrost:
    docker run -d --name bifrost -p 8080:8080 maximhq/bifrost
    @echo "Bifrost running at http://localhost:8080"

bifrost-stop:
    docker stop bifrost && docker rm bifrost

lint:
    uv run ruff check . --fix
    uv run ruff format .

typecheck:
    uv run ty check packages/

check: lint typecheck

edge-structures:
    uv run python examples/edge_cases/pipeline_structures.py

edge-state:
    uv run python examples/edge_cases/state_passing.py

edge-models:
    uv run python examples/edge_cases/multi_model.py

edge-meta:
    uv run python examples/edge_cases/meta_generation.py

edge-all: edge-structures edge-state edge-models edge-meta

test-unit:
    uv run pytest packages/fedotmas/tests/ -v

test-all: test-unit edge-all
