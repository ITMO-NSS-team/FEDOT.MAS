# GAIA Benchmark

Run the [GAIA benchmark](https://huggingface.co/datasets/gaia-benchmark/GAIA) to evaluate FEDOT.MAS multi-agent pipelines.

## Quick start

### Setup

```sh
just venv-gaia
```

Fill in `.env`:

```
HF_TOKEN=hf_...          # Hugging Face token (required to download GAIA)
OPENAI_API_KEY=...       # key/base URL used by the meta-agent
OPENAI_BASE_URL=...
OPENROUTER_API_KEY=sk-or-... # used by GAIA worker agents
SEARXNG_URL=http://localhost:8888  # if SearXNG runs on a non-default port
```

GAIA worker agents use `openai/gpt-5-mini` through OpenRouter by default. Override
with `FEDOTMAS_GAIA_WORKER_MODEL`, `FEDOTMAS_GAIA_WORKER_BASE_URL`, or
`FEDOTMAS_GAIA_WORKER_API_KEY` if needed. The meta-agent model is configured
separately via the usual `FEDOTMAS_META_AGENT_MODEL` / `OPENAI_*` settings.

### Run

```sh
# Single question (quick smoke-test)
just gaia-run run_gaia --split "validation[:1]"

# 10 questions (batch 0 = questions 0–9)
just gaia-batch 0

# 10 questions, difficulty 1 only
just gaia-batch 0 1

# All 165 validation questions
just gaia-all

# Resume from batch 3 to the end
just gaia-resume 3
```

Or directly:

```sh
uv run python benchmarks/gaia/run_gaia.py --difficulty 1 --split "validation[:10]"
```

### Arguments

| Argument | Values | Default |
|---|---|---|
| `--difficulty` | `1`, `2`, `3`, `all` | `all` |
| `--split` | any HF split slice, e.g. `validation[:10]` | `validation[:1]` |

## SearXNG (web search)

Agents use SearXNG for web search. If you already have a SearXNG instance running, set `SEARXNG_URL` in `.env`.

To install and start a local instance:

```sh
just searxng-install
just searxng-start   # runs on http://localhost:18888 by default
```

## Results

Each run saves results under `benchmarks/gaia/gaia_logs/run_<uuid>/`:

```
run_<uuid>/
├── results.json          # metrics + all answers
└── task_<task_id>/
    ├── result.json       # per-task details (answer, ground truth, tokens, elapsed)
    └── leaderboard.json  # {task_id, model_answer} for leaderboard submission
```

System/debug logs: `~/.local/state/fedotmas/log/`.
