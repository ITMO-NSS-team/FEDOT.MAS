# `fedotmas.optimize` — internal module map

User-facing docs: [`docs/optimizer.md`](../../../../../docs/optimizer.md).
This README is for contributors and explains the layout of the package.

The optimizer is an evolutionary loop based on [GEPA](https://arxiv.org/abs/2507.19457)
adapted for `MAWConfig` graphs. Each iteration: pick a parent → mutate via LLM
reflection on a train minibatch → accept if it beats the parent on that minibatch
→ on accept, do a full-val eval and update the Pareto front.

## Files

| File | Role |
|---|---|
| `_optimizer.py` | Public `Optimizer` facade. Wires config, scorer, mutator, selectors, stoppers and delegates to `_engine.run_optimization`. |
| `_engine.py` | The optimization loop itself: iteration scheduling, accept/reject, merge attempts, evaluation dispatch with concurrency, cache lookup, stopper checks. |
| `_config.py` | `OptimizationConfig` dataclass — single source of truth for tunables. |
| `_state.py` | `Candidate`, `Task`, `OptimizationState`, `EvaluationCache`, save/load, plus per-task Pareto helpers (`_per_task_best_set`, `_remove_dominated_programs`). |
| `_strategies.py` | Pluggable selection strategies: `CandidateSelector` (Pareto / best / ε-greedy), `BatchSampler` (epoch-shuffled / random), `ComponentSelector` (all / round-robin). Factories `make_*`. |
| `_scoring.py` | `Scorer` protocol + built-in `LLMJudge` and `ScoringResult`. Custom scorers conform to the protocol. |
| `_stopping.py` | `Stopper` protocol + built-in stoppers (`MaxIterations`, `MaxEvaluations`, `NoImprovement`, `ScoreThreshold`, `CompositeStopper`, `SignalStopper`). |
| `_callbacks.py` | `OptimizationCallback` protocol, `CallbackDispatcher`, and `MetricsCallback` for aggregate stats. |
| `_result.py` | `OptimizationResult` dataclass returned by `Optimizer.optimize`. |
| `_prompts.py` | Reflection / merge LLM prompt templates. Edit here to change instruction-mutation phrasing. |
| `_mutators/` | Mutation operators (see below). |

## `_mutators/` subpackage

| File | Role |
|---|---|
| `_protocol.py` | `Mutator` protocol — anything implementing it is a valid mutator. |
| `_instruction.py` | `InstructionMutator` — the default. Rewrites agent instructions via reflection on minibatch feedback. This is what GEPA does. |
| `_tool.py`, `_model.py`, `_structure.py` | Stubs for future mutation axes (tool assignments, model selection, pipeline topology). Currently scaffolding. |
| `_composite.py` | `CompositeMutator` and `WeightedMutator` — combine multiple mutators with a probability distribution. |

## Where to make common changes

- **Tweak the loop** (eval policy, accept criterion, merge attempt logic) →
  `_engine.py`.
- **Add a new selection / batch / component strategy** → `_strategies.py`,
  then register in the corresponding `make_*` factory.
- **Add a new mutator** → new file under `_mutators/`, conform to
  `_protocol.Mutator`, optionally export from `_mutators/__init__.py`.
- **Add a new stopping condition** → `_stopping.py`, conform to `Stopper`,
  add to the `Optimizer` setup in `_optimizer.py` if it should be on by default.
- **Change the reflection prompt** → `_prompts.py`.
- **Add a new tunable knob** → `_config.py` (and document it in
  `docs/optimizer.md`).
- **Persist new candidate fields** → `_state.py` (`Candidate` dataclass +
  `OptimizationState.save`/`load`).

## Train vs val semantics

Two separate score dicts live on `Candidate`:

- `scores` / `feedbacks` / `states` — **val** results from full-set evals after
  acceptance. Drive Pareto selection, `mean_score`, `best_candidate`.
- `train_scores` / `train_feedbacks` / `train_states` — **train minibatch**
  results. Used only for accept/reject decisions and as reflection examples.

Mixing these caused a regression in the past: train minibatch tasks leaking into
the Pareto frequency-weighting biased selection toward weak parents. Keep them
separate.

## Tests

Live in `packages/fedotmas/tests/optimize/`. Each `_*.py` module has a
corresponding `test_*.py` (e.g. `_state.py` ↔ `test_state.py`).
