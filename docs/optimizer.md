# Optimizer

Evolutionary optimization of multi-agent pipelines. Based on the [GEPA](https://arxiv.org/abs/2507.19457) algorithm adapted for LLM-driven agent graphs.

## How It Works

Agent instructions are treated as "genes" in an evolutionary loop:

1. **Evaluate** the initial (seed) config on a set of tasks
2. **Select** a parent candidate from the population
3. **Mutate** — LLM analyzes execution results and rewrites agent instructions ("reflection")
4. **Evaluate** the mutated candidate on a minibatch of tasks
5. **Accept or reject** — keep the child only if it outperforms the parent
6. **Merge** (optional) — combine instructions from two successful candidates via crossover
7. **Repeat** until convergence or budget exhaustion

## Quick Start

```python
import asyncio
from fedotmas import MAW, Optimizer
from fedotmas.maw.models import MAWAgentConfig, MAWConfig, MAWStepConfig
from fedotmas.optimize import OptimizationConfig, Task

config = MAWConfig(
    agents=[
        MAWAgentConfig(
            name="researcher",
            model="openai/gpt-4o-mini",
            instruction="Research the given topic and provide key facts.",
            output_key="research",
        ),
        MAWAgentConfig(
            name="writer",
            model="openai/gpt-4o-mini",
            instruction="Write a clear summary based on the research.",
            output_key="summary",
        ),
    ],
    pipeline=MAWStepConfig(
        type="sequential",
        children=[
            MAWStepConfig(type="agent", agent_name="researcher"),
            MAWStepConfig(type="agent", agent_name="writer"),
        ],
    ),
)

trainset = [
    Task("Explain quantum computing basics"),
    Task("Describe how neural networks learn"),
    Task("Summarize the history of the internet"),
]

async def main():
    maw = MAW()
    opt = Optimizer(
        maw,
        criteria="Clarity, technical accuracy, and completeness",
        config=OptimizationConfig(
            seed=42,
            max_iterations=10,
            patience=3,
            minibatch_size=2,
        ),
    )

    result = await opt.optimize(trainset, seed_config=config)

    print(f"Best score: {result.best_score:.3f}")
    print(f"Iterations: {result.iterations}")
    print(f"Candidates evaluated: {len(result.all_candidates)}")

    # Use the optimized config
    best_config = result.best_config

asyncio.run(main())
```

- `criteria` — natural-language description of what "good output" means. Passed to the built-in LLM judge.
- `trainset` — tasks the optimizer trains on. Each task is a `Task(input, expected)` named tuple.
- `seed_config` — initial pipeline config. If omitted, generated automatically from the first task.

## Auto-Generated Seed

If you don't have a config yet, skip `seed_config` — the optimizer will generate one from the first task:

```python
result = await opt.optimize(trainset)  # seed_config generated automatically
```

## Train / Validation Split

Pass a separate `valset` to avoid overfitting. Accepted candidates are re-evaluated on the full validation set:

```python
result = await opt.optimize(
    trainset=[Task("task1"), Task("task2"), Task("task3"), Task("task4"), Task("task5")],
    valset=[Task("val1"), Task("val2"), Task("val3")],
    seed_config=config,
)
```

If `valset` is not provided, `trainset` is used for both (with a warning).

## Configuration

```python
from fedotmas.optimize import OptimizationConfig

config = OptimizationConfig(
    # --- Stopping ---
    max_iterations=20,          # hard iteration limit
    patience=5,                 # stop after N iterations without improvement
    score_threshold=0.95,       # stop when score is reached
    max_evaluations=100,        # stop after N total pipeline runs

    # --- Evolution ---
    candidate_selection="pareto",   # "pareto" | "best" | "epsilon_greedy"
    minibatch_size=3,               # tasks per iteration
    batch_strategy="epoch_shuffled",# "epoch_shuffled" | "random"
    use_merge=True,                 # enable crossover
    max_merge_attempts=5,           # max merge operations per run

    # --- LLM temperatures ---
    temperature_reflect=0.7,   # mutation creativity
    temperature_merge=0.5,     # merge creativity
    temperature_judge=0.1,     # scoring consistency (ignored with custom scorer)

    # --- Truncation (optional) ---
    max_state_chars=None,      # limit pipeline state passed to judge. None = no limit
    max_output_chars=None,     # limit agent output in reflection. None = no limit

    # --- Safety ---
    llm_timeout=120.0,             # seconds per LLM call
    max_consecutive_failures=3,    # before emergency agent reshuffle

    # --- Infrastructure ---
    checkpoint_path="state.json",  # save/restore state
    graceful_shutdown=True,        # SIGINT completes current iteration
    seed=42,                       # reproducibility
)
```

### Stopping Criteria

Optimization stops when **any** condition is met:

| Parameter | Default | Description |
|---|---|---|
| `max_iterations` | `20` | Hard iteration limit |
| `patience` | `5` | Iterations without improvement before stopping |
| `score_threshold` | `None` | Target score (disabled by default) |
| `max_evaluations` | `None` | Total pipeline run budget (disabled by default) |
| `graceful_shutdown` | `False` | SIGINT/SIGTERM finishes current iteration instead of aborting |

### Selection Strategies

The `candidate_selection` parameter controls how the parent is chosen for mutation:

| Strategy | Behavior |
|---|---|
| `"pareto"` (default) | Frequency-weighted selection from Pareto front. Candidates dominating more tasks are selected more often |
| `"best"` | Always the highest mean score. Pure exploitation |
| `"epsilon_greedy"` | 90% best, 10% random. Controlled exploration |

### Batch Strategies

| Strategy | Behavior |
|---|---|
| `"epoch_shuffled"` (default) | Shuffled trainset split into sequential minibatches. Every task appears once per epoch |
| `"random"` | Random sample each iteration. No coverage guarantee |

## Expected Answers

Each `Task` can carry an optional expected answer. When provided, the LLM judge uses it as a reference for correctness — no regex parsing needed:

```python
trainset = [
    Task("What is 2+2?", expected="4"),
    Task("Capital of France?", expected="Paris"),
    Task("Explain gravity"),  # no expected — judged by criteria only
]

opt = Optimizer(
    maw,
    criteria="Accuracy and clarity",
    config=OptimizationConfig(max_iterations=10),
)
result = await opt.optimize(trainset, seed_config=config)
```

Tasks **with** `expected` are scored more precisely (the judge sees the reference answer). Tasks **without** `expected` rely on the `criteria` prompt alone.

You can mix both in the same trainset.

## Custom Scorer

By default, the optimizer uses `LLMJudge` — an LLM that scores pipeline output against your `criteria`. You can replace it with any callable matching the `Scorer` protocol:

```python
from fedotmas.optimize import Scorer, ScoringResult, Task

class ExactMatchScorer:
    """Score based on whether the output contains the expected answer."""

    async def evaluate(
        self, task: Task, state: dict) -> ScoringResult:
        output = " ".join(str(v) for v in state.values()).lower()

        if task.expected and task.expected.lower() in output:
            return ScoringResult(score=1.0, feedback="Correct.", reasoning="Match found")

        return ScoringResult(
            score=0.0,
            feedback=f"Expected '{expected}' not found in output.",
            reasoning="No match",
        )

trainset = [
    Task("What is 2+2?", expected="4"),
    Task("Capital of France?", expected="paris"),
]

opt = Optimizer(maw, scorer=ExactMatchScorer(), config=OptimizationConfig(...))
result = await opt.optimize(trainset, seed_config=config)
```

The `Scorer` protocol requires one method:

```python
async def evaluate(self, task: Task, state: dict[str, Any]) -> ScoringResult
```

- `task` — the input task with optional expected answer
- `state` — pipeline output dict (`{agent_output_key: output_value, ...}`)
- Returns `ScoringResult(score=..., feedback=..., reasoning=...)`

`score` must be in `[0.0, 1.0]`. `feedback` is passed to the mutator for reflection. `reasoning` is for logging.

## Callbacks

Track the optimization process by implementing `OptimizationCallback`:

```python
from fedotmas.optimize import OptimizationCallback, MetricsCallback

class PrintCallback(OptimizationCallback):
    def on_iteration_start(self, iteration, state):
        print(f"--- Iteration {iteration} ({len(state.candidates)} candidates) ---")

    def on_candidate_accepted(self, child, parent):
        print(f"  Accepted #{child.index} (score={child.mean_score:.3f})")

    def on_candidate_rejected(self, child, parent):
        print(f"  Rejected #{child.index} (score={child.mean_score:.3f})")

# MetricsCallback collects aggregate statistics
metrics_cb = MetricsCallback()

opt = Optimizer(
    maw,
    criteria="...",
    callbacks=[PrintCallback(), metrics_cb],
    config=OptimizationConfig(...),
)
result = await opt.optimize(trainset, seed_config=config)

# Access metrics
m = metrics_cb.metrics
print(f"Accepted: {m.accepted}, Rejected: {m.rejected}")
print(f"Acceptance rate: {m.acceptance_rate:.1%}")
print(f"Cache hit rate: {m.cache_hit_rate:.1%}")
print(f"Score history: {m.best_score_history}")
```

### Available Hooks

| Method | Trigger |
|---|---|
| `on_iteration_start(iteration, state)` | Start of iteration |
| `on_candidate_evaluated(candidate, tasks)` | After candidate evaluation |
| `on_candidate_accepted(child, parent)` | Child outperforms parent |
| `on_candidate_rejected(child, parent)` | Child does not improve |
| `on_merge_attempted(pair)` | Merge crossover attempted |
| `on_iteration_end(iteration, state)` | End of iteration |
| `on_optimization_end(result)` | Optimization complete |

All methods are optional — override only what you need.

## Merge (Crossover)

When a new candidate is accepted, the optimizer may attempt to merge instructions from two Pareto-front candidates. This is the crossover operator.

Two modes are used automatically:

- **Genealogy merge** — if candidates share a common ancestor, per-agent diff is computed: unchanged instructions are kept as-is, divergent instructions are merged by LLM. More efficient and precise.
- **Direct merge** — if no common ancestor, LLM combines instructions directly.

The merged candidate is accepted only if it scores at least as well as both parents.

```python
config = OptimizationConfig(
    use_merge=True,           # enabled by default
    max_merge_attempts=5,     # max merges per run
    temperature_merge=0.5,    # LLM creativity for merge
)
```

If merge can't find a valid candidate pair (e.g. all pairs are ancestors of each other), the iteration falls through to mutation instead of being wasted.

## Checkpoint / Resume

Save optimization state to resume later:

```python
# Run 1: 5 iterations
opt = Optimizer(
    maw,
    criteria="Quality",
    config=OptimizationConfig(
        checkpoint_path="opt_state.json",
        max_iterations=5,
    ),
)
result = await opt.optimize(trainset, seed_config=config)
print(f"Score after 5 iters: {result.best_score:.3f}")

# Run 2: resume, run up to 15 total
opt = Optimizer(
    maw,
    criteria="Quality",
    config=OptimizationConfig(
        checkpoint_path="opt_state.json",  # same file
        max_iterations=15,
    ),
)
result = await opt.optimize(trainset, seed_config=config)
print(f"Score after 15 iters: {result.best_score:.3f}")
```

The checkpoint includes all candidates, scores, genealogy, and iteration counter.

## Result

```python
result = await opt.optimize(trainset, seed_config=config)

# Best config — ready to use
best_config: MAWConfig = result.best_config
best_score: float = result.best_score

# Iteration stats
result.iterations            # number of iterations completed
result.total_evaluation_runs # total pipeline executions

# Token usage (across judge + mutator + merge)
result.total_prompt_tokens
result.total_completion_tokens

# All candidates with genealogy
for c in result.all_candidates:
    print(f"#{c.index} origin={c.origin} parent={c.parent_index} "
          f"score={c.mean_score:.3f}")

# Pareto front
for c in result.pareto_front():
    print(f"#{c.index} scores={c.scores}")
```

### `Candidate` Fields

| Field | Type | Description |
|---|---|---|
| `index` | `int` | Unique candidate ID |
| `config` | `MAWConfig` | Pipeline configuration |
| `scores` | `dict[str, float]` | Per-task scores (keyed by `task.input`) |
| `mean_score` | `float \| None` | Average across all scored tasks |
| `parent_index` | `int \| None` | Parent candidate (mutation) |
| `merge_parent_indices` | `tuple[int, int] \| None` | Parent candidates (merge) |
| `origin` | `str` | `"seed"`, `"mutation"`, or `"merge"` |
| `on_pareto_front` | `bool` | Whether on the Pareto front |
| `feedbacks` | `dict[str, str]` | Per-task judge feedback |
