from unittest.mock import MagicMock

import pytest
from fedotmas.meta._helpers import format_server_descriptions
from google.adk.utils.instructions_utils import inject_session_state


@pytest.mark.asyncio
async def test_mcp_description_braces_are_not_adk_state_references():
    rendered = format_server_descriptions({"search{tool}": "Search {foo} with {bar}."})
    assert "{tool}" not in rendered
    assert "{foo}" not in rendered
    assert "{bar}" not in rendered
    assert "search〔tool〕" in rendered
    assert "〔foo〕" in rendered
    context = MagicMock()
    context._invocation_context.session.state = {}
    context._invocation_context.artifact_service = None
    assert await inject_session_state(rendered, context) == rendered
