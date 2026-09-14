from __future__ import annotations

from dataclasses import dataclass

import pytest

from fedotmas.mcp import describe as mod
from fedotmas.mcp._config import HttpMCPServer, StdioMCPServer
from fedotmas.mcp.describe import (
    MAX_DESCRIPTION_CHARS,
    MAX_TOOL_DESCRIPTION_CHARS,
    build_description,
    describe_servers,
)


@dataclass
class FakeTool:
    name: str
    description: str = ""


class TestBuildDescription:
    def test_tools_are_named_with_their_first_sentence(self):
        rendered = build_description(
            "srv",
            [
                FakeTool("goto", "Navigate to a URL. Waits for load."),
                FakeTool("markdown", "Extract page content as markdown."),
            ],
        )
        assert rendered == (
            "Tools: goto (Navigate to a URL), "
            "markdown (Extract page content as markdown)."
        )

    def test_a_tool_without_a_description_is_still_named(self):
        assert build_description("srv", [FakeTool("run")]) == "Tools: run."

    def test_a_long_sentence_is_truncated(self):
        rendered = build_description("srv", [FakeTool("x", "y" * 400)])
        assert len(rendered) < MAX_TOOL_DESCRIPTION_CHARS + 40
        assert "…" in rendered

    def test_a_long_tool_list_is_capped_and_counted(self):
        tools = [FakeTool(f"tool_number_{i:02d}", "does a thing") for i in range(40)]
        rendered = build_description("srv", tools)

        assert len(rendered) <= MAX_DESCRIPTION_CHARS + len(", and 40 more.")
        assert "tool_number_00" in rendered
        assert "more." in rendered

    def test_a_single_verbose_tool_is_listed_rather_than_counted(self):
        rendered = build_description("srv", [FakeTool("only", "y" * 4000)])
        assert rendered.startswith("Tools: only (")
        assert "more" not in rendered

    def test_the_count_covers_every_tool_left_out(self):
        tools = [FakeTool(f"t{i:02d}", "w" * 60) for i in range(20)]
        rendered = build_description("srv", tools)
        listed = rendered.count("(")
        assert f"and {20 - listed} more" in rendered

    def test_a_server_with_no_tools_says_so(self):
        assert build_description("srv", []) == (
            "MCP server 'srv', which advertises no tools."
        )


class TestDescribeServers:
    async def test_a_declared_description_is_left_alone(self, monkeypatch):
        async def unexpected(*args, **kwargs):
            raise AssertionError("a described server must not be probed")

        monkeypatch.setattr(mod, "list_server_tools", unexpected)
        registry = {"srv": StdioMCPServer("x", (), description="Does a thing.")}

        described, unreachable = await describe_servers(registry)

        assert described == registry
        assert unreachable == []

    async def test_a_missing_description_is_filled_from_the_tool_list(
        self, monkeypatch
    ):
        async def listing(name, registry=None, *, timeout=None):
            return [FakeTool("search", "Search the web.")]

        monkeypatch.setattr(mod, "list_server_tools", listing)
        registry = {"remote": HttpMCPServer("https://example.org/mcp")}

        described, unreachable = await describe_servers(registry)

        assert described["remote"].description == "Tools: search (Search the web)."
        assert described["remote"].url == "https://example.org/mcp"
        assert unreachable == []

    async def test_an_unreachable_server_is_reported_not_raised(self, monkeypatch):
        async def explode(name, registry=None, *, timeout=None):
            raise ConnectionError("nope")

        monkeypatch.setattr(mod, "list_server_tools", explode)
        registry = {"remote": HttpMCPServer("https://typo.invalid/mcp")}

        described, unreachable = await describe_servers(registry)

        assert [u.name for u in unreachable] == ["remote"]
        assert unreachable[0].reason == "ConnectionError: nope"
        assert described["remote"].description == ""

    async def test_a_timeout_is_reported_plainly(self, monkeypatch):
        async def explode(name, registry=None, *, timeout=None):
            raise TimeoutError

        monkeypatch.setattr(mod, "list_server_tools", explode)

        _, unreachable = await describe_servers(
            {"remote": HttpMCPServer("https://slow.invalid/mcp")}
        )

        assert unreachable[0].reason == "timed out"

    async def test_one_unreachable_server_does_not_hide_a_working_one(
        self, monkeypatch
    ):
        async def listing(name, registry=None, *, timeout=None):
            if name == "bad":
                raise ConnectionError("nope")
            return [FakeTool("ok")]

        monkeypatch.setattr(mod, "list_server_tools", listing)
        registry = {
            "bad": HttpMCPServer("https://typo.invalid/mcp"),
            "good": HttpMCPServer("https://example.org/mcp"),
        }

        described, unreachable = await describe_servers(registry)

        assert [u.name for u in unreachable] == ["bad"]
        assert described["good"].description == "Tools: ok."

    async def test_an_empty_registry_probes_nothing(self, monkeypatch):
        async def unexpected(*args, **kwargs):
            raise AssertionError("nothing to probe")

        monkeypatch.setattr(mod, "list_server_tools", unexpected)

        assert await describe_servers({}) == ({}, [])


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("One. Two.", "One"),
        ("Trailing period.", "Trailing period"),
        ("  collapses   whitespace  ", "collapses whitespace"),
        ("", ""),
    ],
)
def test_first_sentence(text, expected):
    assert mod._first_sentence(text) == expected
