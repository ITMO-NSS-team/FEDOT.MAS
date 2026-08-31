from __future__ import annotations

from pathlib import Path

from fedotmas.mcp._config import DEFAULT_MCP_TIMEOUT_S, StdioMCPServer
from fedotmas.mcp.discovery import discover_local_servers


def _write_toml(tmp_path: Path, name: str, content: str) -> None:
    d = tmp_path / name
    d.mkdir()
    (d / "pyproject.toml").write_text(content)


class TestExternalCommand:
    def test_command_produces_stdio_server(self, tmp_path):
        _write_toml(
            tmp_path,
            "browser",
            """\
[tool.fedotmas]
mcp.name = "browser"
mcp.description = "Headless browser"
mcp.tags = ["web"]
mcp.command = "lightpanda"
mcp.args = ["mcp"]
mcp.timeout = 120
""",
        )
        servers = discover_local_servers(tmp_path)

        assert "browser" in servers
        srv = servers["browser"]
        assert isinstance(srv, StdioMCPServer)
        assert srv.command == "lightpanda"
        assert srv.args == ("mcp",)
        assert srv.timeout == 120
        assert srv.description == "Headless browser"
        assert srv.tags == ("web",)

    def test_command_without_args(self, tmp_path):
        _write_toml(
            tmp_path,
            "simple",
            """\
[tool.fedotmas]
mcp.name = "simple"
mcp.command = "my-binary"
""",
        )
        servers = discover_local_servers(tmp_path)

        srv = servers["simple"]
        assert srv.command == "my-binary"
        assert srv.args == ()
        assert srv.timeout == DEFAULT_MCP_TIMEOUT_S

    def test_command_skips_project_scripts(self, tmp_path):
        """When mcp.command is set, [project.scripts] is not required."""
        _write_toml(
            tmp_path,
            "ext",
            """\
[tool.fedotmas]
mcp.name = "ext"
mcp.command = "ext-bin"
""",
        )
        servers = discover_local_servers(tmp_path)
        assert "ext" in servers
        assert servers["ext"].command == "ext-bin"


class TestScriptsFallback:
    def test_scripts_still_works(self, tmp_path):
        _write_toml(
            tmp_path,
            "classic",
            """\
[tool.fedotmas]
mcp.name = "classic"
mcp.description = "A classic server"

[project.scripts]
mcp-classic = "mcp_classic:main"
""",
        )
        servers = discover_local_servers(tmp_path)

        assert "classic" in servers
        srv = servers["classic"]
        assert isinstance(srv, StdioMCPServer)
        assert "run" in srv.args
        assert srv.description == "A classic server"

    def test_no_command_no_scripts_skipped(self, tmp_path):
        _write_toml(
            tmp_path,
            "broken",
            """\
[tool.fedotmas]
mcp.name = "broken"
""",
        )
        servers = discover_local_servers(tmp_path)
        assert "broken" not in servers


class TestTimeoutResolution:
    """A declared timeout wins; otherwise the env var, otherwise the default."""

    def _scan(self, tmp_path, extra: str = ""):
        _write_toml(
            tmp_path,
            "classic",
            f"""\
[tool.fedotmas]
mcp.name = "classic"
{extra}

[project.scripts]
mcp-classic = "mcp_classic:main"
""",
        )
        return discover_local_servers(tmp_path)["classic"]

    def test_default_when_unset(self, tmp_path, monkeypatch):
        monkeypatch.delenv("FEDOTMAS_MCP_TIMEOUT_S", raising=False)
        assert self._scan(tmp_path).timeout == DEFAULT_MCP_TIMEOUT_S

    def test_env_overrides_default(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FEDOTMAS_MCP_TIMEOUT_S", "600")
        assert self._scan(tmp_path).timeout == 600

    def test_declared_timeout_wins_over_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FEDOTMAS_MCP_TIMEOUT_S", "600")
        assert self._scan(tmp_path, "mcp.timeout = 45").timeout == 45

    def test_unparsable_env_falls_back(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FEDOTMAS_MCP_TIMEOUT_S", "soon")
        assert self._scan(tmp_path).timeout == DEFAULT_MCP_TIMEOUT_S

    def test_non_positive_env_falls_back(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FEDOTMAS_MCP_TIMEOUT_S", "0")
        assert self._scan(tmp_path).timeout == DEFAULT_MCP_TIMEOUT_S
