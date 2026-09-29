from __future__ import annotations

import json

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from mcp_rubber_recipe_predictor.server import mcp


@pytest.mark.anyio
async def test_predict_tool_is_registered() -> None:
    async with Client(mcp) as client:
        tools = await client.list_tools()
    assert [tool.name for tool in tools] == ["predict_rubber_properties"]


@pytest.mark.anyio
async def test_predict_tool_returns_properties_for_unchanged_recipe() -> None:
    recipe = {
        "nr_smr20_phr": 55.0,
        "sbr1502_phr": 45.0,
        "carbon_black_n220_phr": 55.0,
        "zinc_oxide_phr": 5.0,
        "stearic_acid_phr": 2.0,
        "tmq_antioxidant_phr": 1.5,
        "antiozonant_6ppd_phr": 1.5,
        "process_oil_phr": 5.0,
        "sulfur_phr": 5.0,
        "tmtd_accelerator_phr": 2.0,
        "reclaim_phr": 5.0,
    }
    async with Client(mcp) as client:
        result = await client.call_tool("predict_rubber_properties", recipe)
    payload = json.loads(result.content[0].text)
    assert payload["status"] == "prediction_completed"
    assert payload["recipe_input_unchanged"] == {
        **{key: value for key, value in recipe.items() if key != "antiozonant_6ppd_phr"},
        "6ppd_antiozonant_phr": 1.5,
    }
    assert set(payload["predicted_properties"]) == {
        "thermal_conductivity_w_mk",
        "oil_swelling_pct_1006h",
        "water_swelling_pct_1006h",
        "specific_gravity",
    }
    assert payload["provenance"]["source_rows"] == 20


@pytest.mark.anyio
async def test_predict_tool_rejects_recipe_outside_dataset_conditions() -> None:
    recipe = {
        "nr_smr20_phr": 55.0,
        "sbr1502_phr": 45.0,
        "carbon_black_n220_phr": 55.0,
        "zinc_oxide_phr": 4.0,
        "stearic_acid_phr": 2.0,
        "tmq_antioxidant_phr": 1.5,
        "antiozonant_6ppd_phr": 1.5,
        "process_oil_phr": 5.0,
        "sulfur_phr": 5.0,
        "tmtd_accelerator_phr": 2.0,
        "reclaim_phr": 5.0,
    }
    async with Client(mcp) as client:
        with pytest.raises(ToolError, match="open dataset holds these ingredients fixed"):
            await client.call_tool("predict_rubber_properties", recipe)
