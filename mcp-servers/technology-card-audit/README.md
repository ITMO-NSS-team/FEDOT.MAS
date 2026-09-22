# Technology card audit MCP server

This server supports the FEDOT.MAS technological-card demo. It exposes two
deterministic tools:

- `read_technology_card` returns quantitative norms and exact references from
  a compact demonstration TTK;
- `audit_historical_productivity` compares those norms with a synthetic
  historical execution slice using explicit threshold rules.

The fixture is intentionally labeled as synthetic. It demonstrates the
integration and report contract without claiming access to the private STAIRS
or SAMPO dumps described in `DEMOS.md`.
