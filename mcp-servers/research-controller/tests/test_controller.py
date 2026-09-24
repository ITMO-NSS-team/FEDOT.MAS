from mcp_research_controller.controller import ResearchController


def test_research_state_persists_and_merges_between_updates():
    controller = ResearchController()

    first = controller.update_research_state(
        research_id="paper-lookup",
        goal="Find the identifier reported in the paper",
        findings=["The paper title is known"],
        sources_checked=["https://example.org/paper"],
        unresolved_questions=["Which identifier is reported?"],
    )
    second = controller.update_research_state(
        research_id="paper-lookup",
        goal="Find the identifier reported in the paper",
        findings=["The paper title is known", "The methods section is available"],
        evidence=["Methods section names the identifier"],
        sources_checked=["https://example.org/paper"],
        unresolved_questions=[],
        confidence=0.86,
    )

    assert first["counts"]["findings"] == 1
    assert second["counts"] == {
        "findings": 2,
        "evidence": 1,
        "sources_checked": 1,
        "unresolved_questions": 0,
        "failed_attempts": 0,
        "search_queries": 0,
    }
    assert controller.get_next_action("paper-lookup")["decision"] == "synthesize"


def test_controller_response_is_compact():
    controller = ResearchController()
    controller.update_research_state(
        goal="Compare two public product release dates",
        findings=["A finding " * 30],
        evidence=["Evidence " * 30],
        sources_checked=["https://example.org/source"],
        unresolved_questions=["Which date is correct?"],
    )

    response = controller.get_next_action()

    assert response["decision"] == "continue_search"
    assert len(str(response)) < 500
    assert "Evidence " not in str(response)


def test_repeated_semantic_search_intent_recommends_strategy_change():
    controller = ResearchController()
    controller.update_research_state(
        goal="Find an identifier in a source",
        search_queries=[
            "exact title enzyme identifier",
            "enzyme identifier search by paper title",
            "paper title search for enzyme identifiers",
        ],
        unresolved_questions=["What identifier does the source report?"],
    )

    response = controller.get_next_action()

    assert response["decision"] == "change_strategy"
    assert "same semantic intent" in response["reason"]


def test_repeated_failed_strategy_recommends_strategy_change():
    controller = ResearchController()
    for attempt in (
        "search: exact title returned no useful result",
        "search: title variant returned no useful result",
        "search: alternate title returned no useful result",
    ):
        controller.update_research_state(
            goal="Find an identifier in a source",
            failed_attempts=[attempt],
            unresolved_questions=["What identifier does the source report?"],
        )

    response = controller.get_next_action()

    assert response["decision"] == "change_strategy"
    assert "Repeated failures" in response["reason"]


def test_summarized_failed_attempt_count_recommends_strategy_change():
    controller = ResearchController()
    controller.update_research_state(
        goal="Find a source that identifies a named entity",
        failed_attempts=["search: exact title variants found no answer 5 times"],
        unresolved_questions=["Which identifier is reported?"],
    )

    response = controller.get_next_action()

    assert response["decision"] == "change_strategy"


def test_repeated_identical_query_is_counted_as_a_search_loop():
    controller = ResearchController()
    for _ in range(3):
        controller.update_research_state(
            goal="Find a reported identifier",
            search_queries=["find the reported identifier"],
        )

    response = controller.get_next_action()

    assert response["decision"] == "change_strategy"
    assert "same semantic intent" in response["reason"]


def test_sufficient_evidence_recommends_synthesis():
    controller = ResearchController()
    controller.update_research_state(
        goal="Compare release dates for two software libraries",
        evidence=["Official changelog A", "Official changelog B"],
        sources_checked=["https://example.org/a", "https://example.org/b"],
        unresolved_questions=[],
    )

    response = controller.get_next_action()

    assert response["decision"] == "synthesize"
    assert "no gaps remain" in response["reason"]


def test_different_research_goals_require_separate_ids():
    controller = ResearchController()
    controller.update_research_state(
        research_id="release-date",
        goal="Find a software release date",
    )

    try:
        controller.update_research_state(
            research_id="release-date",
            goal="Find a different source",
        )
    except ValueError as exc:
        assert "use a new id" in str(exc)
    else:
        raise AssertionError("Expected a new research id for a different goal")
