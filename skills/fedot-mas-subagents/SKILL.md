---
name: fedot-mas-subagents
description: Create and run bounded multi-agent workflows with FEDOT.MAS, including host-native Codex or Claude subagents when the requested model has no OpenAI-compatible endpoint. Use for FEDOT.MAS decomposition, subagent orchestration, or paired single-agent versus multi-agent evaluations; do not use for ordinary one-agent tasks that have no useful decomposition.
---

# FEDOT.MAS subagents

Use FEDOT.MAS as the workflow authority: preserve its pipeline, state keys, and
execution order even when the language-model calls are fulfilled by the host's
native subagent tool.

## Choose the execution path

- Use native FEDOT.MAS execution when every requested model is reachable through
  a supported provider or OpenAI-compatible endpoint. Build and run `MAW` (fixed
  workflow) or `MAS` (LLM routing) directly.
- Use the host bridge when Codex or Claude can spawn the requested model but no
  API endpoint is available. Read [host-bridge.md](references/host-bridge.md),
  then run [fedot_host_bridge.py](scripts/fedot_host_bridge.py). The bridge uses
  FEDOT.MAS/ADK for the real workflow and exchanges each LLM turn through JSON
  queue files.
- Do not silently substitute a model or claim that the bridge is a native API
  integration. If the exact requested model cannot be selected by the host,
  report that before running.

## Design the workflow

Prefer the smallest decomposition that creates a distinct information or review
boundary. Typical useful roles are independent researcher/solver, critic or
verifier, and final synthesizer. A worker that merely repeats the same prompt is
overhead, not a decomposition.

Create a `MAWConfig` JSON with:

- unique agent names and `output_key` values;
- an explicit provider-prefixed model label for every agent, such as
  `host/model-id` in bridge mode;
- state references such as `{research?}` only to outputs available earlier in
  the workflow;
- a final agent whose output is the deliverable;
- bounded loops (`max_iterations`) and a proportional agent/call budget.

Read [config-patterns.md](references/config-patterns.md) only when a sequential,
parallel, or review-loop example is needed. Validate the config with
`scripts/validate_config.py` before execution.

## Execute and verify

Give each subagent only the task data, prior outputs, tools, and permissions its
role needs. In evaluations, never expose reference answers, hidden tests, or the
other arm's output. Use the same model, task input, reasoning setting, and
stopping conditions across comparison arms; record call count, latency, and
token or character estimates separately from provider billing data.

Keep the generated config, queue transcript, final state, and verification
result. Report negative results as-is: a correct intermediate output does not
count as an improvement if the final scorer is unchanged.
