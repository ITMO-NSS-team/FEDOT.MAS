# Evaluation verdict

## Outcome

FEDOT.MAS can represent the requested recipe-generation prototype as a dynamic
master-orchestrator `MAS`. The current result is stored under
`recipe_prediction`; the current state does not contain `final_report`.

Tested upstream revision:
`7d1378edf54dc87b5c57f58e7361d9cf58e92120` (2026-09-06).

## What was demonstrated

- A 20-row open-data table was reconstructed from the published 5 x 4
  SBR/NR/N220 experimental grid.
- A deterministic degree-2 polynomial ridge surface generated one candidate
  recipe and four property predictions.
- The user input contains only the specification. The master invokes the
  discovered `predict_rubber_recipe` MCP tool and receives the numerical JSON
  during the MAS run.
- Each leave-one-out estimate refits on 19 of 20 rows.
- The candidate lies inside the published formulation domain and satisfies all
  four requested point constraints.
- A real FEDOT.MAS/Google ADK coordinator tree calls the MCP predictor, injects
  its result into three specialist requests, and restores control to the master
  after every call.
- All current agent labels are `host/gpt-5.6-terra`; the saved Terra turns are
  displayed in a persistent ADK Web session.

## Execution boundary

`gpt-5.6-terra` was invoked by the Codex subscription host because no
OpenAI-compatible provider endpoint is exposed for this subscription model.
The recorded responses are replayed by the real FEDOT.MAS/ADK topology for GUI
inspection, while the local MCP predictor executes normally. Consequently, this
validates the MAS structure, MCP integration, state propagation, routing
transcript, and result presentation, but the Terra turns are not native
LiteLLM/provider-API inference calls.

## Scientific limitations

The source values were digitized from plots and the dataset has only 20 rows.
LOOCV measures internal interpolation error and is not external validation. All
four point constraints pass, but after applying a conservative one-RMSE margin,
only specific gravity remains robustly inside its bound. The proposed compound
must undergo mixing, rheometry, vulcanization, and independent property testing.
No claim is made about strength, abrasion, wet grip, rolling resistance,
durability, production readiness, or tire safety.
