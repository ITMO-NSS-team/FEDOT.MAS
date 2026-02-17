from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

# Default LLM model for all agents (meta-agent and pipeline agents).
# Format examples:
#   "gemini-2.5-flash"                         — Gemini (native, no extra deps)
#   "claude-sonnet-4-20250514"                  — Claude (requires anthropic pkg)
#   "openrouter/anthropic/claude-sonnet-4"      — via LiteLLM
DEFAULT_MODEL: str = os.getenv("FEDOTMAS_DEFAULT_MODEL", "gemini-2.5-flash")

# Model used specifically for the meta-agent (pipeline generator).
# Falls back to DEFAULT_MODEL if not set.
META_AGENT_MODEL: str = os.getenv("FEDOTMAS_META_AGENT_MODEL", "") or DEFAULT_MODEL

# Temperature for meta-agent generation (lower = more deterministic pipeline configs).
META_AGENT_TEMPERATURE: float = float(
    os.getenv("FEDOTMAS_META_AGENT_TEMPERATURE", "0.3")
)

# Maximum iterations for LoopAgent nodes when not specified in config.
DEFAULT_MAX_LOOP_ITERATIONS: int = int(
    os.getenv("FEDOTMAS_DEFAULT_MAX_LOOP_ITERATIONS", "3")
)
