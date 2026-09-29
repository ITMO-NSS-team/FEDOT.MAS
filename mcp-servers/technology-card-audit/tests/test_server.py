from __future__ import annotations

import json

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from mcp_technology_card_audit.server import CARD_ID, mcp


@pytest.mark.anyio
async def test_tools_are_registered() -> None:
    async with Client(mcp) as client:
        tools = await client.list_tools()
    assert [tool.name for tool in tools] == [
        "read_technology_card",
        "audit_historical_productivity",
    ]


@pytest.mark.anyio
async def test_card_norms_are_internally_consistent() -> None:
    async with Client(mcp) as client:
        result = await client.call_tool("read_technology_card", {"card_id": CARD_ID})
    payload = json.loads(result.content[0].text)
    norms = payload["norms"]
    assert norms["crew_productivity_per_shift"]["value"] == 8.0
    assert norms["crew_size"]["value"] / norms["crew_productivity_per_shift"]["value"] == 0.5
    assert payload["source_kind"] == "synthetic_demo_fixture"


@pytest.mark.anyio
async def test_audit_returns_only_threshold_violations() -> None:
    async with Client(mcp) as client:
        result = await client.call_tool(
            "audit_historical_productivity",
            {"card_id": CARD_ID, "upper_multiplier": 2.0, "lower_divisor": 3.0},
        )
    payload = json.loads(result.content[0].text)
    assert payload["records_checked"] == 6
    assert payload["violations_count"] == 3
    assert [row["object"] for row in payload["violations"]] == ["Куст 18", "Куст 7", "Куст 24"]
    assert [row["fact_rate"] for row in payload["violations"]] == [20.0, 2.25, 25.0]
    assert all(row["norm_rate"] == 8.0 for row in payload["violations"])
    assert all(row["card_ref"] for row in payload["violations"])


@pytest.mark.anyio
async def test_unknown_card_is_rejected() -> None:
    async with Client(mcp) as client:
        with pytest.raises(ToolError, match="Unknown demo card"):
            await client.call_tool("read_technology_card", {"card_id": "missing"})


@pytest.mark.anyio
async def test_custom_thresholds_are_used_in_violation_labels() -> None:
    async with Client(mcp) as client:
        result = await client.call_tool("audit_historical_productivity", {
            "card_id": CARD_ID, "upper_multiplier": 2.5, "lower_divisor": 2.0,
        })
    payload = json.loads(result.content[0].text)
    assert payload["violations_count"] == 2
    assert {row["fact_rate"] for row in payload["violations"]} == {2.25, 25.0}
    assert {row["violation_type"] for row in payload["violations"]} == {
        "выше нормы более чем в 2.5 раза", "ниже нормы более чем в 2 раза",
    }
