# research-controller MCP

`research-controller` keeps a compact, in-memory ledger for a research workstream
and recommends whether to continue searching, change strategy, or synthesize. It
does not browse or retrieve evidence.

Research agents call `update_research_state` periodically, especially before more
expensive searches. Include new `search_queries`, `findings`, `evidence`,
`sources_checked`, the current `unresolved_questions`, `failed_attempts`, and
optional confidence and remaining budget. Send only new `search_queries` and
`failed_attempts` events on each update so repetitions can be detected. Then call
`get_next_action` and follow its compact recommendation. Findings, evidence, and
sources merge across updates; unresolved questions describe the current gaps. Use
a stable, distinct `research_id` for each workstream.

The controller identifies repeated query intent with lightweight token overlap,
recommends changing strategy after repeated failures or a high query count, and
recommends synthesis when evidence is sufficient. State lasts for the lifetime of
the MCP process and is bounded in memory.
