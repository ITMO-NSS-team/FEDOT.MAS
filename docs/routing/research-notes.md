# Research Notes — LLM Routing (as of 2026-05-21)

Background literature and critical assessment behind the routing module design. Companion to `plan.md`.

## The six families of LLM routing

The 2026 landscape stabilised around six approaches:

| Family | Idea | Representative methods |
|---|---|---|
| Difficulty-aware | Small classifier estimates query complexity → escalation | BEST-Route, ICL-Router |
| Preference-aligned | Train router on human/LLM-judge preferences | RouteLLM (ICLR 2025), Arch-Router (Jun 2025) |
| Clustering-based | k-means partitioning + per-cluster best LLM | UniRoute |
| RL / bandit (online) | Contextual bandit or RL-trained policy | MixLLM, PILOT, Router-R1, dueling-feedback bandits (Oct 2025) |
| Uncertainty-based | Route by model confidence / conformal prediction | CP-Router |
| Cascade | Sequential escalation with quality estimator | FrugalGPT, AutoMix, Cascade Routing (+4% on RouterBench) |

Our module is **memory-based with online updates**, closest to the RL/bandit family but using retrieval over an experience store instead of a parametric policy. This is the EvoRoute style.

## Key empirical finding — LLMRouterBench (Jan 2026, 400K instances, 33 models)

1. **Many routing methods do not reliably beat simple baselines** under unified evaluation. Includes some commercial routers.
2. **Embeddings have little impact on routing performance.** Most embedding-based work is over-engineered.
3. **Diminishing returns from larger model pools.** Curated 3–5 models often beat large ensembles. Pool curation matters as much as routing.

Implications we acted on:
- Don't make the embedding signal load-bearing. Our retrieval uses *union* (role ∪ semantic ∪ tools), not *intersection* — role and tool overlap can match without any embedding signal.
- Pool config is explicit and small. No "throw everything in" defaults.

## EvoRoute (arxiv 2601.02695, our architectural inspiration)

**Authors:** Zhang et al., submitted 2026-01-06, v1, no peer review. First author Guibin Zhang also did MasRouter (Feb 2025, the direct multi-agent precursor).

**Method (what we adopted):**

- Step-level records `(agent role, LLM, instruction, embedding, tools, cost, duration, success_step, success_task)`.
- Retrieval = union of role match, semantic similarity (`θ = 0.85`), tool overlap.
- Pareto filter on means → Thompson sampling with Normal-Inverse-Gamma priors → utility `w_p·perf − w_c·cost − w_d·delay` with weights (1.0, 0.1, 0.05).
- Cold-start: ~480 records seeded via uniform-random sampling on 50 TaskCraft tasks ($28.80 + 3.6h).

**Reported numbers:** −80% cost, −70% latency on GAIA / BrowseComp+ when integrated into CK-Pro / Smolagents frameworks, with comparable or improved performance.

### Why we do not treat the reported numbers as a target

- **Baselines are weak.** Compared only to MAS-specific routers (PromptLLM, GraphRouter, MasRouter) and to "always-Claude-4" defaults. No comparison against RouteLLM, Arch-Router, or FrugalGPT — the actual SOTA from non-MAS routing.
- "−80% vs Claude-4" is what any reasonable router should achieve; not evidence over best-in-class.
- Only 2 MAS frameworks tested (CK-Pro, Smolagents). Authors acknowledge.
- **Code was promised at `github.com/bingreeky/evo-route` and is still 404 as of 2026-08-19.** No public artefact to reproduce or audit.

### Why we adopt the design anyway

- **Method is principled, not ad-hoc.** kNN retrieval, Pareto, Thompson sampling, NIG posteriors — all standard components, integrated coherently.
- **Reimplementable from the paper.** All formulas and default hyperparameters are stated.
- **Author credibility.** Guibin Zhang also wrote MasRouter (with code released), so the team understands the field.
- **Ablations are honest.** The cold-start budget is disclosed (-13.21% performance if removed); multi-faceted retrieval contributes -11.22%; Thompson sampling -6.52%; Pareto filtration only -1.52%.

## What we deliberately chose differently from EvoRoute

- **Embeddings via OpenRouter (`text-embedding-3-small`), not local `all-MiniLM-L6-v2`.** No 80 MB + torch dependency; uses the existing transport.
- **Non-informative (Jeffreys) NIG priors + `+∞` exploration sentinel for n<2.** Sidesteps the prior-tuning problem when metrics have wildly different scales (perf 0-1, cost USD, delay seconds). EvoRoute doesn't specify priors explicitly.
- **Randomised tie-break among `+∞` utility candidates** — found this bug on the first integration test; without it, under-explored models are starved.
- **`LlmPool` warns on zero prices** — silently inert cost dimension would be a footgun in production.

## Open questions for the future

- Is per-LLM-call granularity better than per-agent-turn for our pipeline depth? EvoRoute defaults to per-call; we follow. Worth re-evaluating once we have data.
- Would a learned router (e.g. Arch-Router) beat memory-based at our scale? Probably not given LLMRouterBench's finding, but the experimental setup would be straightforward once Phase 5 lands.
- Cold-start strategy beyond uniform random: epsilon-greedy with current best? UCB instead of TS? Defer until we see Phase 5 numbers.
