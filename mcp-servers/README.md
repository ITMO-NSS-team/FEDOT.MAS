## MCP servers

FEDOT.MAS uses [MCP](https://modelcontextprotocol.io/) (Model Context Protocol) to give agents access to external tools: file downloads, web scraping, code execution, etc.

When a user describes a task, the **meta-agent** sees the list of all registered MCP servers with their descriptions. It decides which tools each pipeline agent needs and includes them in the generated pipeline config. At runtime the pipeline builder launches the required servers as subprocesses via `uv run --directory` and connects to them over stdio.

### Auto-discovery

Discovery triggers when `mcp_servers` is `"all"` or a list of server names
(e.g. `["download-url-content"]`). The registry walks up from the `fedotmas`
package to find the workspace root (a `pyproject.toml` with
`[tool.uv.workspace]`), then scans `mcp-servers/*/pyproject.toml` for
`[tool.fedotmas.mcp]` sections. Default `mcp_servers=None` means no servers,
no discovery.

When installed from PyPI (outside a workspace), discovery finds no root and
returns an empty registry. To use servers in this case, either pass a
`dict[str, MCPServerConfig]` directly, or point `discover_servers()` at any
directory with the same layout:

```python
from fedotmas import MAS
from fedotmas.mcp import discover_servers

mas = MAS(mcp_servers=discover_servers("/path/to/my-servers"))
```

### Adding a new MCP server

1. Create a directory under `mcp-servers/`:

```
mcp-servers/my-server/
  pyproject.toml
  src/
    my_server/
      server.py
```

2. In `pyproject.toml`, add the standard project metadata **and** a `[tool.fedotmas.mcp]` section:

```toml
[project]
name = "my-mcp-server"
version = "0.1.0"
dependencies = ["fastmcp>=2.14.5"]

[project.scripts]
my-mcp-server = "my_server.server:main"

[tool.fedotmas.mcp]
name = "my-server"
description = "Short description of what the server does — the meta-agent reads this to decide when to use it."
tags = ["relevant", "tags"]
```

3. On next `MAS(mcp_servers="all")` call, the server appears in the registry and becomes available to the meta-agent.

The `name` field is the key used in pipeline configs (e.g. `"tools": ["my-server"]`). The entry point is taken from the first key in `[project.scripts]`. Each server has its own `uv`-managed virtualenv created on first run.
