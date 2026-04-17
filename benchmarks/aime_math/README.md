# AIME Math Benchmark

Evaluation of FEDOT.MAS on AIME competition math problems. Setup
follows the GEPA paper
([arxiv.org/abs/2507.19457](https://arxiv.org/abs/2507.19457)) for
comparable results.

## Datasets

- **Train / val**: `AI-MO/aimo-validation-aime` (AIME 2022-2024, 90 problems), shuffled and split 50/50 → 45 train + 45 val.
- **Test**: `MathArena/aime_2025` (AIME 2025, 30 problems).

All answers are integers in the range 0-999. The scorer
(`scorer.py`) extracts the final integer from free-form solver output
(supports `\boxed{N}`, `"answer is N"`, or the trailing number).

## Install

```bash
pip install -r benchmarks/aime_math/requirements.txt
```

Set required env vars for the solver model:

- `OPENAI_API_KEY` (or provider-specific key)
- `OPENAI_BASE_URL` (e.g. `https://openrouter.ai/api/v1` for OpenRouter)

## Run

### Baseline (no optimization)

Evaluates the seed prompt on the test set only, skipping the optimizer:

```bash
python benchmarks/aime_math/run.py --max-iterations 0
```

### Full optimization run

```bash
python benchmarks/aime_math/run.py \
    --max-evaluations 500 --test-repeats 5 \
    2>&1 | tee logs/aime_$(date +%Y%m%d_%H%M%S).log
```

### Skip baseline test evaluation (use known accuracy)

```bash
python benchmarks/aime_math/run.py \
    --max-evaluations 500 --test-repeats 5 \
    --skip-baseline-test --baseline-accuracy 0.465
```

### Quick debug run

```bash
python benchmarks/aime_math/run.py \
    --max-iterations 2 \
    --train-limit 3 --val-limit 3 --test-limit 3
```

## CLI flags

| Flag | Default | Description |
|---|---|---|
| `--max-iterations` | 10000 | Optimizer iterations. `0` skips optimization entirely. |
| `--max-evaluations` | `None` | Stop after N total evaluation runs (rollout budget). |
| `--seed` | 42 | RNG seed for dataset shuffle and optimizer. |
| `--output-dir` | `outputs/aime_math` | Where to write result JSON. |
| `--train-limit` | `None` | Cap trainset size (debug). |
| `--val-limit` | `None` | Cap valset size (debug). |
| `--test-limit` | `None` | Cap testset size (debug). |
| `--test-repeats` | 1 | Repeat each test question N times. |
| `--concurrency` | 8 | Parallel task evaluations (semaphore limit). |
| `--max-output-tokens` | `None` | Cap completion tokens per LLM call. |
| `--skip-baseline-test` | `False` | Skip baseline evaluation on test set. |
| `--baseline-accuracy` | `None` | Use this value as baseline accuracy instead of evaluating. |

All flags also work as env vars with `AIME_` prefix, e.g. `AIME_SOLVER_MODEL="qwen/qwen3-8b"`.

## Solver model

Change the solver model via env:

```bash
AIME_SOLVER_MODEL="openai/gpt-4.1-mini" python benchmarks/aime_math/run.py --max-iterations 0
AIME_SOLVER_MODEL="qwen/qwen3-8b"       python benchmarks/aime_math/run.py --max-iterations 0
```

Non-OpenAI models are routed through `_ProxyClient` using `OPENAI_BASE_URL` (e.g. OpenRouter).

## Reference results

| Model | Accuracy | Time | Cost |
|---|---|---|---|
| `gpt-4.1-mini` baseline (16 runs, 480 trials) | 46.5% ± 4.5% | ~5 min/run | ~$0.19/run |
| `qwen/qwen3-8b` baseline (4 runs averaged) | 63.3% | ~27 min/run | ~$0.22/run |

GEPA paper baselines: `gpt-4.1-mini` = 49.33%, `qwen3-8b` = 27.33%.
