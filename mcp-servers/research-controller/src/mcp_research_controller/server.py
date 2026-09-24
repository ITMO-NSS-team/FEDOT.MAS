"""MCP tools for maintaining compact research state and choosing next actions."""

from __future__ import annotations

from fastmcp import FastMCP

from mcp_research_controller.controller import ResearchController

mcp = FastMCP("research-controller")
_controller = ResearchController()


@mcp.tool
def update_research_state(
    goal: str,
    research_id: str = "default",
    findings: list[str] | None = None,
    evidence: list[str] | None = None,
    sources_checked: list[str] | None = None,
    unresolved_questions: list[str] | None = None,
    failed_attempts: list[str] | None = None,
    search_queries: list[str] | None = None,
    confidence: float | None = None,
    remaining_budget: float | None = None,
) -> dict[str, object]:
    """Update the research ledger with new evidence, searches, gaps, and budget.

    Findings, evidence, and sources merge across updates. Search queries and failed
    attempts are event lists: send only new entries so repeats can be counted.
    The unresolved question list is a current snapshot and may be empty.
    """
    return _controller.update_research_state(
        goal=goal,
        research_id=research_id,
        findings=findings,
        evidence=evidence,
        sources_checked=sources_checked,
        unresolved_questions=unresolved_questions,
        failed_attempts=failed_attempts,
        search_queries=search_queries,
        confidence=confidence,
        remaining_budget=remaining_budget,
    )


@mcp.tool
def get_next_action(research_id: str = "default") -> dict[str, object]:
    """Recommend whether to continue, change strategy, or synthesize findings."""
    return _controller.get_next_action(research_id)


def main() -> None:
    mcp.run(show_banner=False)
