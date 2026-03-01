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
