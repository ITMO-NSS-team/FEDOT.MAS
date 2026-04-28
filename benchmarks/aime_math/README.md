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
| `--concurrency` | 8 | Parallel evals during baseline/optimized testset eval. |
| `--eval-concurrency` | 8 | Parallel evals inside the optimizer loop. |
| `--max-output-tokens` | `None` | Cap completion tokens per LLM call. |
| `--skip-baseline-test` | `False` | Skip baseline evaluation on test set. |
| `--baseline-accuracy` | `None` | Use this value instead of evaluating baseline. |
| `--eval-best-on-train` | `False` | *(debug)* See note below. |
| `--checkpoint-path` | `None` | Save/resume optimizer state. See note below. |

All flags also work as env vars with `AIME_` prefix, e.g. `AIME_SOLVER_MODEL="qwen/qwen3-8b"`.

**`--eval-best-on-train`** *(debug)* — after optimization, evaluates the best
candidate on the full trainset (+45 evals) to surface train/val/test gaps.
Val accuracy is reused from the optimizer's full-val eval (no extra cost).
Useful for diagnosing whether a flat test result is due to distribution
shift (`train ≈ val ≪ test`) or selection bias on the val-fold (`train < val`).

**`--checkpoint-path`** — path to a JSON file where the optimizer writes its
state after every iteration. If the file already exists at startup, the run
resumes from it (skipping completed iterations and reusing cached evals).
Useful for long runs where a crash, OOM, or `Ctrl+C` would otherwise lose
hours of work. Pass the same `--seed` on resume so the train/val split and
batch sequence stay consistent.

## Solver model

Change the solver model via env:

```bash
AIME_SOLVER_MODEL="openai/gpt-4.1-mini" python benchmarks/aime_math/run.py --max-iterations 0
AIME_SOLVER_MODEL="qwen/qwen3-8b"       python benchmarks/aime_math/run.py --max-iterations 0
```

Non-OpenAI models are routed through `_ProxyClient` using `OPENAI_BASE_URL` (e.g. OpenRouter).

## Reference results

### Baselines (no optimization)

| Model | Accuracy | Time | Cost |
|---|---|---|---|
| `gpt-4.1-mini` (16 runs, 480 trials) | 46.5% ± 4.5% | ~5 min/run | ~$0.19/run |
| `qwen/qwen3-8b` (4 runs averaged) | 63.3% | ~27 min/run | ~$0.22/run |
| `gpt-oss-120b` (5 repeats, 150 trials) | 74.7% | — | — |

### Optimization (baseline → optimized on test)

| Model | Seed config | Budget | Baseline | Optimized | Improvement | Time | Cost |
|---|---|---|---|---|---|---|---|
| `gpt-4.1-mini` (ours) | manual | 500 evals | 46.5% | — | — | — | — |
| `gpt-oss-120b` (ours) | manual | 1839 evals | 74.7% | 82.0% | +7.3 pp | 19.8 h | $4.29 |
| `gpt-oss-120b` (ours) | FEDOT.MAS | 2000 evals | 74.7% | 80.0% | +5.3 pp | 21.8 h | $5.12 |
| `gpt-4.1-mini` (GEPA paper) | manual | 1839 evals | 49.33% | 59.44% | +10.11 pp | — | — |
| `qwen3-8b` (GEPA paper) | manual | 1839 evals | 27.33% | 32.00% | +4.67 pp | — | — |

**Seed config** column distinguishes the source of the initial pipeline
configuration that the optimizer evolves from:

- *manual* — single-agent pipeline with a hand-written instruction.
  GEPA paper rows use the same hand-written instruction we use in our
  `manual` rows.
- *FEDOT.MAS* — pipeline generated by the FEDOT.MAS meta-agent
  (`generate_seed.py`) from an abstract task description, with no
  concrete problem examples passed to the generator.

Both `gpt-oss-120b` rows use `--seed 7` and the same train/val/test
split. The FEDOT.MAS-seeded run reached val accuracy 0.911 (vs val
baseline 0.867) over 188 optimizer iterations, demonstrating that
meta-agent-generated seed configs are a viable starting point for
GEPA-style instruction optimization.
