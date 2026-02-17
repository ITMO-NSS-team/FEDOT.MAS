# AGENTS.md

## Code Quality

All generated Python code must be checked before committing:
```bash
# Type checking
uv run ty check src/

# Linting and formatting
uv run ruff check src/
uv run ruff format --check src/
```

Fix any errors before submitting code. Do not suppress warnings without justification.
