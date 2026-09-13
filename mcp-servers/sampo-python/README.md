# mcp-sampo-python

Leakage-safe general Python workspace for the public SAMPO benchmark.

Set `E2B_API_KEY` in `.env`. The server creates an isolated E2B workspace with
outbound network disabled and uploads only:

- `benchmark_inputs.csv`
- `allowed_target_labels.csv`

It exposes persistent Python execution and validated prediction export. It has
no tool for reading or uploading arbitrary host files.

Generate current structural-review configurations:

```bash
uv run python scripts/run_sampo_phase_4_1.py
```
