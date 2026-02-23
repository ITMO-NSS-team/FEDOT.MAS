## MCP servers

FEDOT.MAS uses [MCP](https://modelcontextprotocol.io/) (Model Context Protocol) to give agents access to external tools: file downloads, web scraping, code execution, etc.

When a user describes a task, the **meta-agent** sees the list of all registered MCP servers with their descriptions. It decides which tools each pipeline agent needs and includes them in the generated pipeline config. At runtime the pipeline builder launches the required servers as subprocesses via `uv run --directory` and connects to them over stdio.

### Auto-discovery

MCP servers are discovered automatically at import time. The registry scans `mcp-servers/*/pyproject.toml` for a `[tool.fedotmas.mcp]` section and registers every server it finds.

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

3. On next import of `fedotmas.mcp`, the server appears in the registry and becomes available to the meta-agent.

The `name` field is the key used in pipeline configs (e.g. `"tools": ["my-server"]`). The entry point is taken from the first key in `[project.scripts]`. Each server has its own `uv`-managed virtualenv created on first run.
