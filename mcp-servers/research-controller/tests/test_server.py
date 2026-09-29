import json
import uuid

import pytest
from fastmcp import Client
from mcp_research_controller.server import mcp


@pytest.mark.anyio
async def test_mcp_tools_persist_state_and_return_compact_recommendations():
    research_id = f"release-date-{uuid.uuid4()}"
    async with Client(mcp) as client:
        updated = await client.call_tool(
            "update_research_state",
            {
                "research_id": research_id,
                "goal": "Compare public release dates for two software libraries",
                "evidence": ["Official sources report matching release dates"],
                "evidence_urls": [
                    "https://vendor.example/a",
                    "https://archive.example/b",
                ],
                "required_fields": ["release_date"],
                "filled_fields": ["release_date"],
                "unresolved_questions": [],
            },
        )
        update_data = json.loads(updated.content[0].text)
        recommendation = await client.call_tool(
            "get_next_action",
            {"research_state": update_data["research_state"]},
        )
        recommendation_data = json.loads(recommendation.content[0].text)
        followed_update = await client.call_tool(
            "update_research_state",
            {
                "goal": "Compare public release dates for two software libraries",
                "research_state": recommendation_data["research_state"],
                "last_recommendation_followed": True,
            },
        )

    followed_data = json.loads(followed_update.content[0].text)
    assert update_data["counts"]["evidence"] == 1
    assert recommendation_data["action"] == "synthesize"
    assert len(recommendation.content[0].text) < 2500
    telemetry = followed_data["research_state"]["telemetry"]
    assert telemetry["followed_recommendations"] == 1
    assert telemetry["intervention_outcomes"][-1]["followed"] is True
