venv:
    uv venv
    uv pip install -e packages/fedotmas -e packages/fedotmas-synapse
    uv run python -c "import shutil, os; shutil.copy('.env.example', '.env') if not os.path.exists('.env') else None"

venv-dev: venv
    uv sync --group dev
    uv run prek install
    @echo "Dev environment ready"

bifrost:
    docker run -d --name bifrost -p 8080:8080 maximhq/bifrost
    @echo "Bifrost running at http://localhost:8080"

bifrost-stop:
    docker stop bifrost && docker rm bifrost

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
