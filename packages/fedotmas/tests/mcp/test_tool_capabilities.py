from __future__ import annotations

from types import SimpleNamespace

from fedotmas.maw.builder import build
from fedotmas.maw.models import MAWAgentConfig, MAWConfig, MAWStepConfig
from fedotmas.mcp import ToolCapability, create_toolset, tool_capability
from fedotmas.mcp._config import StdioMCPServer


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

    assert tool_filter(SimpleNamespace(name="search")) is True
    assert tool_filter(SimpleNamespace(name="telemetry")) is False
    assert tool_filter(SimpleNamespace(name="websearch_tavily_telemetry")) is False
    assert "telemetry" in toolset._fedotmas_filtered_diagnostic_tools
    assert "websearch_tavily_telemetry" in toolset._fedotmas_filtered_diagnostic_tools


def test_runtime_can_request_unfiltered_diagnostic_tools_internally():
    toolset = create_toolset(
        "websearch-tavily",
        registry={
            "websearch-tavily": StdioMCPServer(command="echo", args=()),
        },
        include_diagnostic=True,
    )

    assert toolset.tool_filter is None
