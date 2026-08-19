# Per-call LLM Routing — Implementation Plan

A standalone, opt-in plugin that picks an LLM per call inside a FEDOT.MAS pipeline based on accumulated experience. Inspired by [EvoRoute](https://arxiv.org/abs/2601.02695); see `research-notes.md` for why we picked this design.

Branch: `dev/models-routing`.

**Status as of 2026-08-19: paused.** Phases 0–3 are done and merged into the branch; Phase 4 was dropped and Phase 5 is wired but not yet run. Work is parked in favour of other priorities, not blocked on anything technical.

## Design invariants

1. **Opt-in.** When the plugin is not registered, `MAWAgentConfig.model` continues to drive static per-agent selection. Zero breaking changes to MAW.
2. **Algorithm core is ADK-free.** `packages/fedotmas/src/fedotmas/routing/` has no imports from `google.adk`. Only the plugin glues it to the pipeline.
3. **Pool config lives on the plugin constructor,** not in `MAWConfig` — routing is a routing concern, not a pipeline concern.
4. **Per-LLM-call granularity.** Each LLM call gets its own routing decision and its own record in the experience store. `success_task` is backfilled by `trace_id` after the task is scored.

## Phase status

| Phase | Status | Deliverable | Effort |
|---|---|---|---|
| 0 — ADK mutability spike | ✅ done | regression guard at `packages/fedotmas/tests/test_llm_request_mutation.py` | 0.5d |
| 1 — Routing core + store | ✅ done | `packages/fedotmas/src/fedotmas/routing/` + 59 unit tests | 2d |
| 2 — Plugin integration | ✅ done | `LLMRoutingPlugin` at `packages/fedotmas/src/fedotmas/plugins/_routing.py` + 13 integration tests | 1d |
| 3 — Success attribution | ✅ done | `ControlledRun.invocation_id` + `commit_task_score` wired into both benchmark runners | 0.5d |
| 4 — Cold-start + back-fill | ⛔ dropped | superseded by cold-start in `Router._random_pick` | — |
| 5 — GAIA comparison | ⏸ wired, not run | `--routing-pool` flag in `examples/gaia/run_gaia.py` | 0.5d |

## Phase 0 — ADK mutability spike (done)

**Goal:** confirm Plan-A (direct mutation of `llm_request.model` inside `before_model_callback`) is viable for the plugin design.

**Outcome:** Plan-A confirmed. `LiteLlm.generate_content_async` reads `effective_model = llm_request.model or self.model` per call, and `LlmRequest` is a non-frozen pydantic model, so in-place mutation propagates to the transport.

**Artifact:** `packages/fedotmas/tests/test_llm_request_mutation.py` — control test (no mutation → original model) + main test (mutation → new model). Kept as long-lived regression guard against future ADK changes.

## Phase 1 — Routing core + experience store (done)

**Goal:** all the algorithmic + persistence machinery, isolated from ADK and from the plugin.

**Subpackage `packages/fedotmas/src/fedotmas/routing/`:**

- `models.py` — `LlmPoolEntry`, `LlmPool` (warns on zero prices), `Weights`, `ExperienceRecord`, `LlmStats`, `RoutingDecision`.
- `algorithm.py` — `aggregate`, `pareto_filter`, `thompson_sample`, `score_utility`. Non-informative NIG priors; `EXPLORE_UTILITY = +∞` sentinel forces exploration when a model has fewer than `MIN_N_FOR_INFERENCE = 2` records in retrieval.
- `store.py` — `ExperienceStore` Protocol + `SQLiteExperienceStore`. Retrieval is union of (agent-role match, semantic similarity ≥ θ, tool overlap), bounded by `max_pool` (default 10 000) with `ORDER BY id DESC`.
- `embeddings.py` — async caching wrapper around `litellm.aembedding`. Default model `openai/text-embedding-3-small` via OpenRouter (existing transport, no new API key).
- `selector.py` — `Router` orchestrating one decision; returns `RoutingOutcome(decision, query_embedding)` so the plugin reuses the embedding when persisting the record.

**Tests:** 59 unit tests in `packages/fedotmas/tests/routing/`.

## Phase 2 — Plugin integration (done)

`LLMRoutingPlugin(BasePlugin)` at `packages/fedotmas/src/fedotmas/plugins/_routing.py`, exported via `fedotmas.plugins`.

**Hooks implemented:**

- `before_run_callback(invocation_context)` — reset per-trace step counter at `invocation_id`.
- `before_model_callback(callback_context, llm_request)` — extract `(agent_name, tools)` and compose the retrieval query as `{system_instruction}\n\n{user_text}`, call `await router.select(...)`, mutate `llm_request.model`, stash `_Pending(t0, outcome, query, tools, original_model)` keyed by `(trace_id, agent_name)`. Workflow-prefixed agents (`seq_/par_/loop_`) are skipped. On selection error, fall through without mutating.
- `after_model_callback(callback_context, llm_response)` — compute `cost` from `llm_response.usage_metadata` × pool pricing; `duration = now - t0`; append `ExperienceRecord` with `success_step = 1.0`; increment per-trace `step_idx`.
- `on_model_error_callback` — append with `success_step = 0.0` and `cost = 0.0`.

**Public API:**

- `LLMRoutingPlugin(*, pool, store=None, embedder=None, weights=None, cold_start_threshold=50, sim_threshold=0.85, db_path="outputs/routing/experience.sqlite", rng=None)`
- `plugin.commit_task_score(trace_id, score) -> int` (Phase 3 hook).
- `plugin.store` / `plugin.router` accessors for tests and back-fill scripts.

**Test coverage** (`packages/fedotmas/tests/plugins/test_routing_plugin.py`, 13 tests):

- `before_model_callback` mutates to a pool model, skips workflow nodes, falls through on router failure.
- `after_model_callback` writes a record with correct trace/agent/query/cost, uses pool pricing, increments `step_idx`, no-ops when before was skipped.
- `on_model_error_callback` writes a record with `success_step=0.0`.
- Parallel agents keyed correctly by `(trace_id, agent_name)`.
- `commit_task_score` backfills and returns rows updated; unknown trace returns 0.
- `tools_dict` keys persisted sorted.

## Phase 3 — Success attribution (done)

**Problem:** `success_task` is only known after the benchmark scorer runs, which happens *outside* the ADK pipeline (in the benchmark runner, after `Controller.run()` returns). So no built-in ADK callback fires at the right time.

**What landed:**

1. `run_pipeline` captures `invocation_id` from the first yielded event and exposes it on `PipelineResult.invocation_id`. On the error path (raised `RuntimeError`), the id is attached as an attribute on the exception so callers can read it.
2. `ControlledRun.invocation_id: str | None` is propagated from both the success path (`PipelineResult.invocation_id`) and the error path (`getattr(exc, "invocation_id", None)`) — in both `Controller._execute` and `IterableRun._run`.
3. `LLMRoutingPlugin.commit_task_score(trace_id, score) -> int` was already in place from Phase 2.
4. Benchmark runners (`benchmarks/hotpot_qa/run.py`, `benchmarks/aime_math/run.py`) now thread an optional `routing_plugin: LLMRoutingPlugin | None` through `main` → `evaluate_on` → `_solve_one`. When supplied, the plugin is registered for the run and `commit_task_score(run.invocation_id, score)` fires after scoring — including on failed pipelines (score=0.0).

**Tests:**

- `tests/control/test_controller.py::test_run_surfaces_invocation_id_on_{success,error}` — Controller propagation.
- `tests/plugins/test_routing_plugin.py::TestPhase3BackfillBatch::test_three_tasks_two_agents_each` — end-to-end: 3 traces × 2 agent steps, mix of success and mid-pipeline failure; asserts every record gets `success_task` populated and the failed trace lands at `0.0`.

## Phase 4 — Langfuse back-fill (dropped)

Originally planned as a `backfill_langfuse.py` script that would synthesise `ExperienceRecord`s
from historical Langfuse `generation` observations. **Dropped 2026-05-29** after checking the
premise, which did not hold:

1. **Langfuse holds no trace-level scores.** `LangfusePlugin` never calls `lf.score(trace_id, ...)`,
   and no benchmark wires the plugin in — only `examples/langfuse_test.py` uses it. The existing
   history is therefore ad-hoc and unscored.
2. **Records with `success_task = NULL` are worse than no records at all.** `aggregate()` filters
   unscored rows out of the posterior, but `store.count()` still counts them. Back-filling would
   push the store past `cold_start_threshold`, switching the random-pick safety net off, while
   contributing zero usable signal — every candidate lands on the `+∞` explore sentinel and the
   choice degenerates to a tie-break.
3. **A real back-fill is three changes, not a script:** write `lf.score()` from the benchmark
   runners, wire `LangfusePlugin` into hotpot/aime/gaia, and only then back-fill — and it would be
   useful only for traces recorded after the first two land.

**What remains, and is sufficient:** the cold-start half of this phase was already implemented in
Phase 1. `Router._random_pick` returns a uniform-random model while `store.count() <
cold_start_threshold` (default 50). At GAIA's ~3 LLM calls per task that budget is spent in roughly
17 tasks, so a 100-task run has ample exploited history.

Cheap optional follow-up if this is ever revisited: a one-line
`lf.score(trace_id, score, name="task_success")` next to each `commit_task_score` call, so scored
history starts accumulating from now on.

## Phase 5 — GAIA comparison (wired, not yet run)

**Target changed from HotpotQA to GAIA (2026-05-29).** HotpotQA and AIME pipelines are structurally
too simple for routing to show an effect — one or two agents over a single tool set, so nearly every
call looks alike to the retriever. GAIA's `examples/gaia/run_gaia.py` runs `maw.run()` with
`mcp_servers="all"`: multi-tool, multi-step, and the meta-agent generates a fresh pipeline per task.
That variety is what the experience store is supposed to exploit.

**Prerequisite, resolved.** Because GAIA pipelines differ per task, agent-role matching is a weak
retrieval key and embedding similarity carries the load — which made the missing `system_instruction`
in the embedding query a blocker rather than a nicety. `_routing.py` now composes the query as
`{system_instruction}\n\n{user_text}` via `_system_instruction_text` / `_compose_query`. ADK's
`instructions` request processor runs before `before_model_callback`, so the instruction is already
state-substituted when we read it.

**How it is wired.** `run_gaia.py` takes `--routing-pool <path.json>`. When set, one shared
`LLMRoutingPlugin` is built from the pool file and passed into every task's `MAW`, and the task score
is committed with `commit_task_score(maw.last_result.invocation_id, 1.0 if is_correct else 0.0)`.
When unset, the runner behaves exactly as before, on static `MAWAgentConfig.model`. The shipped
`examples/gaia/routing_pool.json` is a three-model pool: `openai/gpt-4.1-mini`, `openai/gpt-4.1`,
`openai/o4-mini`.

**The experiment, when it runs:** the same GAIA subset twice, baseline vs routed, compared on
`(accuracy, total_cost, total_duration)`. A routing summary in `after_run_callback` showing a
per-agent model-choice histogram is still to be added.

**Exit criteria:** the routed variant reaches either ≥ baseline accuracy at lower cost, or higher
accuracy at ≤ 1.2× baseline cost. If neither, tune the `(w_p, w_c, w_d)` weights — that is a
research-iteration loop, not a phase failure.

## Cross-cutting risks

1. **Cost-table accuracy.** Per-token prices are user-supplied. Stale pricing silently biases routing. Mitigation: pricing is documented as user responsibility; consider future LiteLLM lookup (out of scope for v1).
2. **Concurrency.** Parallel agents → concurrent `before_model_callback` calls. SQLite is in WAL mode + per-instance lock; per-trace state in the plugin should be keyed by `(trace_id, agent_name)` not just `agent_name` to avoid races.
3. **Privacy.** Records persist query text and instruction. Future `store_text: bool` flag would let users persist only embedding + metadata.

## Known limitations

Identified in review and consciously deferred. None of them breaks correctness for a benchmark run;
all of them matter before the plugin is enabled in a long-running process.

**Before production use:**

- **No cleanup when a run raises.** ADK calls `after_run_callback` as a plain sequential call, not
  from a `finally`, so a failed run leaves its per-call state in the plugin. A FIFO cap of 1000
  entries bounds the leak but does not close it. A `plugin.reset()` from the runner's `finally`
  would.
- **Synchronous lock in an async store.** `SQLiteExperienceStore` guards `append`/`retrieve`/`count`
  with a `threading.Lock`, which blocks the event loop. `asyncio.Lock` or `asyncio.to_thread()`
  would unblock parallel agents.
- **Retrieval is a full scan.** `retrieve` loads up to `max_pool` rows, decodes every embedding and
  computes cosine similarity in Python — O(N) per LLM call. Fine at benchmark scale (~5k records);
  past that it needs a vector index.

**Smaller:**

- Default `db_path` is relative (`outputs/routing/experience.sqlite`), so it follows the CWD instead
  of the repo root.
- `ExperienceRecord.id` is `None` at construction and mutated by `store.append`.
- `plugin.store` / `plugin.router` expose internals; convenient for tests and scripts, but not a
  stable public API.
- Jeffreys priors with `MIN_N_FOR_INFERENCE = 2` give very wide early posteriors, so Thompson
  sampling can flap in the first few dozen decisions — most visibly on the cost dimension.
- The plugin reads two private ADK attributes, isolated to the `_invocation_id()` / `_agent_name()`
  helpers. It will break if ADK renames them; the fix is one line in each.
- `benchmarks/{hotpot_qa,aime_math}/run.py` accept a `routing_plugin` argument but their CLIs do not
  expose it — reachable from Python, not from the shell. `run_gaia.py` has the `--routing-pool`
  flag.
