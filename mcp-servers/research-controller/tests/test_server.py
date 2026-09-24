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
                "evidence": [
                    "Official release notes for library A",
                    "Official notes for B",
                ],
                "sources_checked": ["https://example.org/a", "https://example.org/b"],
                "unresolved_questions": [],
            },
        )
        recommendation = await client.call_tool(
            "get_next_action", {"research_id": research_id}
        )

    update_data = json.loads(updated.content[0].text)
    action_data = json.loads(recommendation.content[0].text)
    assert update_data["counts"]["evidence"] == 2
    assert action_data["decision"] == "synthesize"
    assert len(recommendation.content[0].text) < 500
