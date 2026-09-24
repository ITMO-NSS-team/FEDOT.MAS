import json

from mcp_research_controller.controller import ResearchController


def _update(
    controller: ResearchController,
    research_state: dict | None = None,
    **kwargs,
) -> dict:
    return controller.update_research_state(
        research_state=research_state,
        goal=kwargs.pop("goal", "Find a public product release date"),
        **kwargs,
    )["research_state"]


def test_state_snapshot_persists_across_controller_instances():
    original_controller = ResearchController()
    snapshot = _update(
        original_controller,
        research_id="release-date",
        findings=["Release date is listed in the official notes"],
        evidence=["Official release notes identify the date"],
        evidence_urls=[
            "https://vendor.example/releases",
            "https://archive.example/item",
        ],
        required_fields=["release_date", "product"],
        filled_fields=["release_date"],
        unresolved_questions=["Which product version?"],
    )
    encoded = json.dumps(snapshot)

    restarted_controller = ResearchController()
    result = restarted_controller.get_next_action(json.loads(encoded))

    assert result["action"] == "continue_search"
    assert result["research_state"]["goal"] == "Find a public product release date"
    assert result["research_state"]["unresolved_questions"] == [
        "Which product version?"
    ]


def test_recommendations_have_structured_action_and_reason():
    controller = ResearchController()
    snapshot = _update(
        controller,
        required_fields=["release_date"],
        unresolved_questions=["What is the release date?"],
    )

    result = controller.get_next_action(snapshot)

    assert result["action"] == "continue_search"
    assert result["reason"] == "unresolved_questions"
    assert result["guidance"]


def test_similar_queries_escalate_from_change_strategy_to_strategy_blocked():
    controller = ResearchController()
    snapshot = _update(
        controller,
        search_queries=["exact title enzyme identifiers"],
        unresolved_questions=["Which identifier is reported?"],
    )
    snapshot = _update(
        controller,
        snapshot,
        search_queries=["enzyme ID in the paper methods"],
    )
    early = controller.get_next_action(snapshot)
    assert early["action"] == "change_strategy"
    assert early["reason"] == "repeated_low_value_search"

    snapshot = _update(
        controller,
        early["research_state"],
        search_queries=["enzyme identifier from article"],
    )
    blocked = controller.get_next_action(snapshot)

    assert blocked["action"] == "strategy_blocked"
    assert blocked["reason"] == "repeated_low_value_search"
    assert "does not stop other research" in blocked["guidance"]
    assert blocked["research_state"]["telemetry"]["strategy_blocked_events"] == 1


def test_repeated_failed_strategy_triggers_strategy_blocked():
    controller = ResearchController()
    snapshot = _update(
        controller,
        failed_attempts=[
            "search: title variant found no source",
            "search: alternate title found no source",
            "search: exact title found no source",
        ],
        unresolved_questions=["Which source contains the value?"],
    )

    result = controller.get_next_action(snapshot)

    assert result["action"] == "strategy_blocked"
    assert result["reason"] == "repeated_low_value_search"


def test_evidence_sufficiency_uses_sources_fields_and_gaps_not_confidence():
    controller = ResearchController()
    snapshot = _update(
        controller,
        evidence=["Both sources report the same release date"],
        evidence_urls=[
            "https://vendor.example/releases",
            "https://archive.example/releases",
        ],
        required_fields=["release_date", "product_version"],
        filled_fields=["release_date", "product_version"],
        unresolved_questions=[],
        confidence=0.25,
    )

    result = controller.get_next_action(snapshot)

    assert result["action"] == "synthesize"
    assert result["reason"] == "evidence_requirements_satisfied"


def test_high_confidence_does_not_override_missing_fields_or_sources():
    controller = ResearchController()
    snapshot = _update(
        controller,
        evidence=["A candidate value was found"],
        evidence_urls=["https://one.example/item"],
        required_fields=["release_date", "product_version"],
        filled_fields=["release_date"],
        unresolved_questions=[],
        confidence=0.99,
    )

    result = controller.get_next_action(snapshot)

    assert result["action"] == "continue_search"
    assert result["reason"] == "evidence_requirements_incomplete"


def test_telemetry_survives_json_round_trip_with_follow_through_and_search_counts():
    controller = ResearchController()
    snapshot = _update(
        controller,
        search_queries=[
            "exact enzyme identifier",
            "enzyme id source",
            "enzyme identifier source",
        ],
        unresolved_questions=["Which source reports the identifier?"],
    )
    recommendation = controller.get_next_action(snapshot)
    intervention = _update(
        ResearchController(),
        recommendation["research_state"],
        search_queries=["publisher archive source identifier"],
        last_recommendation_followed=True,
    )
    serialized = json.dumps(intervention)
    restored = json.loads(serialized)
    telemetry = restored["telemetry"]
    event = telemetry["recommendations"][-1]
    outcome = telemetry["intervention_outcomes"][-1]

    assert telemetry["controller_calls"] == 3
    assert telemetry["recommendation_count"] == 1
    assert telemetry["strategy_blocked_events"] == 1
    assert telemetry["followed_recommendations"] == 1
    assert event["searches_before"] == 3
    assert event["searches_after"] == 4
    assert event["followed"] is True
    assert outcome["searches_before"] == 3
    assert outcome["searches_after"] == 4
    assert outcome["followed"] is True
