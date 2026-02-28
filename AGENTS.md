# AGENTS.md

## Code Quality

All generated Python code must be checked before committing:
```bash
# Type checking
uv run ty check packages/

# Linting and formatting
uv run ruff check packages/
uv run ruff format --check packages/
```

Fix any errors before submitting code. Do not suppress warnings without justification.
