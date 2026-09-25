from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

from fedotmas.maw.builder import build
from fedotmas.maw.models import MAWAgentConfig, MAWConfig, MAWStepConfig
from fedotmas.mcp import ToolCapability, create_toolset, tool_capability
from fedotmas.mcp._config import StdioMCPServer
from mcp import ListToolsResult, Tool


def test_shared_capability_map_covers_research_tools():
    expected = {
        "websearch_tavily__search": ToolCapability.DISCOVERY,
        "searxng_search": ToolCapability.DISCOVERY,
        "web_scraping_markdown": ToolCapability.URL_INSPECTION,
        "document_read_document": ToolCapability.DOCUMENT_INSPECTION,
        "download_file": ToolCapability.DOCUMENT_INSPECTION,
        "youtube-transcript__get_video_info": ToolCapability.MEDIA_INSPECTION,
        "get_transcript": ToolCapability.MEDIA_INSPECTION,
        "get_timed_transcript": ToolCapability.MEDIA_INSPECTION,
        "complete_browser_task": ToolCapability.BROWSER_NAVIGATION,
        "solve_with_code": ToolCapability.COMPUTATION,
        "tavily_telemetry": ToolCapability.DIAGNOSTIC,
    }

    assert {name: tool_capability(name) for name in expected} == expected


def test_bare_non_web_search_is_not_web_discovery():
    assert tool_capability("search") == ToolCapability.OTHER
    assert (
        tool_capability("search", description="Search the web for sources")
        == ToolCapability.DISCOVERY
    )
    assert tool_capability("search", server="pubchem") == ToolCapability.OTHER


def test_tavily_worker_toolset_exposes_search_and_filters_telemetry():
    registry = {
        "websearch-tavily": StdioMCPServer(command="echo", args=()),
    }
    config = MAWConfig(
        agents=[
            MAWAgentConfig(
                name="researcher",
                instruction="Search for the requested source.",
                output_key="research",
                tools=["websearch-tavily"],
            )
        ],
        pipeline=MAWStepConfig(type="agent", agent_name="researcher"),
    )

    agent = build(config, mcp_registry=registry, autonomous=False)
    toolset = agent.tools[0]
    tool_filter = toolset.tool_filter

    assert tool_filter(SimpleNamespace(name="search"), None) is True
    assert tool_filter(SimpleNamespace(name="telemetry"), None) is False
    assert (
        tool_filter(SimpleNamespace(name="websearch_tavily_telemetry"), None) is False
    )
    assert "telemetry" in toolset._fedotmas_filtered_diagnostic_tools
    assert "websearch_tavily_telemetry" in toolset._fedotmas_filtered_diagnostic_tools


async def test_tavily_tools_load_through_adk_and_filter_telemetry(monkeypatch):
    toolset = create_toolset(
        "websearch-tavily",
        registry={"websearch-tavily": StdioMCPServer(command="echo", args=())},
    )
    tools_response = ListToolsResult(
        tools=[
            Tool(name="search", inputSchema={"type": "object"}),
            Tool(name="telemetry", inputSchema={"type": "object"}),
        ]
    )
    monkeypatch.setattr(
        toolset, "_execute_with_session", AsyncMock(return_value=tools_response)
    )

    tools = await toolset.get_tools()

    assert [tool.name for tool in tools] == ["search"]
    assert toolset._fedotmas_filtered_diagnostic_tools == {"telemetry"}


def test_runtime_can_request_unfiltered_diagnostic_tools_internally():
    toolset = create_toolset(
        "websearch-tavily",
        registry={
            "websearch-tavily": StdioMCPServer(command="echo", args=()),
        },
        include_diagnostic=True,
    )

    assert toolset.tool_filter is None
