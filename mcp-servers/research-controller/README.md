# research-controller MCP

`research-controller` evaluates a compact, explicit JSON snapshot owned by the
caller. Pass the returned `research_state` into each subsequent call and preserve
it in FEDOT session state or the standard research output/handoff. The MCP process
does not own the source of truth. It recommends whether to continue searching,
change strategy, block a repeated strategy, or synthesize. It does not browse or
retrieve evidence.

Research agents call `update_research_state` before expensive work, after several
searches or failures, and before final synthesis or handoff. Include new
`search_queries`, `findings`, `evidence`, `evidence_urls`, `independent_sources`,
`required_fields`, `filled_fields`, `sources_checked`, current
`unresolved_questions`, `failed_attempts`, and optional budget/confidence. Send
only new query and failure events so loops can be counted. Evidence sufficiency
defaults to one source; set `min_sources` or `require_independent_sources` when
the task itself needs a stronger standard. Call `get_next_action`
with the returned snapshot, follow its action, and carry the resulting snapshot
forward. On the next update, report `last_recommendation_followed` as a boolean.

`strategy_blocked` means stop repeating the current strategy, try another evidence
source/tool, or synthesize current findings; it does not stop other work. Synthesis
requires no unresolved questions, all reported required fields filled, evidence,
and the configured minimum source count. Telemetry in
`research_state.telemetry` records calls, recommendations, blocked events, follow
through, and search counts before/after recommendations. Event histories are
bounded; aggregate counters remain in the snapshot.
