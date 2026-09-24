"""MCP tools for maintaining compact research state and choosing next actions."""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from mcp_research_controller.controller import ResearchController

mcp = FastMCP("research-controller")
_controller = ResearchController()


@mcp.tool
def update_research_state(
    goal: str,
    research_state: dict[str, Any] | None = None,
    research_id: str | None = None,
    findings: list[str] | None = None,
    evidence: list[str] | None = None,
    evidence_urls: list[str] | None = None,
    independent_sources: list[str] | None = None,
    sources_checked: list[str] | None = None,
    required_fields: list[str] | None = None,
    filled_fields: list[str] | None = None,
    unresolved_questions: list[str] | None = None,
    failed_attempts: list[str] | None = None,
    search_queries: list[str] | None = None,
    confidence: float | None = None,
    remaining_budget: float | None = None,
    last_recommendation_followed: bool | None = None,
) -> dict[str, Any]:
    """Update the research ledger with new evidence, searches, gaps, and budget.

    Pass the ``research_state`` object returned by the previous controller call.
    The updated snapshot is returned for the caller to store in FEDOT session state.
    Send only new search queries and failed attempts; unresolved questions are a
    current snapshot. Report whether the previous recommendation was followed.
    """
    return _controller.update_research_state(
        goal=goal,
        research_state=research_state,
        research_id=research_id,
        findings=findings,
        evidence=evidence,
        evidence_urls=evidence_urls,
        independent_sources=independent_sources,
        sources_checked=sources_checked,
        required_fields=required_fields,
        filled_fields=filled_fields,
        unresolved_questions=unresolved_questions,
        failed_attempts=failed_attempts,
        search_queries=search_queries,
        confidence=confidence,
        remaining_budget=remaining_budget,
        last_recommendation_followed=last_recommendation_followed,
    )


@mcp.tool
def get_next_action(research_state: dict[str, Any]) -> dict[str, Any]:
    """Evaluate the supplied state and return action, reason, guidance, and state."""
    return _controller.get_next_action(research_state)


def main() -> None:
    mcp.run(show_banner=False)
