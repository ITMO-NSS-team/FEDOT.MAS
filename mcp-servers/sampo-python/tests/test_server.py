from __future__ import annotations

import pytest

from mcp_sampo_python.server import _validate_predictions, mcp


async def test_server_exposes_only_safe_workspace_tools():
    names = {tool.name for tool in await mcp.list_tools()}
    assert names == {"prepare_public_workspace", "run_code", "save_predictions"}


def test_prediction_validation_rejects_unknown_labels():
    content = "example_id,top_1,top_2,top_3\n1,not a label,not a label,not a label\n"
    with pytest.raises(ValueError, match="allowed labels"):
        _validate_predictions(content)
