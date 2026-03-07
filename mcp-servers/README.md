## MCP servers

FEDOT.MAS uses [MCP](https://modelcontextprotocol.io/) (Model Context Protocol) to give agents access to external tools like file downloads, web scraping, and code execution.

When a user describes a task, the **meta-agent** reads the descriptions of all registered MCP servers and decides which tools each pipeline agent needs. At runtime the pipeline builder wraps each server in an ADK `McpToolset`. The ADK then either launches a subprocess (for stdio servers) or opens an HTTP/SSE connection (for remote ones).

### Auto-discovery

Discovery runs when you pass `mcp_servers="all"` or a list of server names (e.g. `["download-url-content"]`). The registry walks up from the `fedotmas` package to find the workspace root (a `pyproject.toml` with `[tool.uv.workspace]`), then scans every `mcp-servers/*/pyproject.toml` for `[tool.fedotmas.mcp]` sections.

If you leave `mcp_servers` at the default `None`, no discovery happens and no servers are registered.

If `fedotmas` is installed from PyPI (outside a workspace), discovery finds no root and returns an empty registry. You can still use servers by passing a config dict directly or by pointing `discover_servers()` at a directory with the same layout:

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

2. Add the standard project metadata **and** a `[tool.fedotmas.mcp]` section to `pyproject.toml`:

```toml
[project]
name = "my-mcp-server"
version = "0.1.0"
dependencies = ["fastmcp>=2.14.5"]

[project.scripts]
my-mcp-server = "my_server.server:main"

[tool.fedotmas.mcp]
name = "my-server"
description = "Short description of what the server does. The meta-agent reads this to decide when to use it."
tags = ["relevant", "tags"]
timeout = 120  # optional, seconds to wait for MCP session (default: 60)
```

3. Next time you call `MAS(mcp_servers="all")`, the new server appears in the registry and the meta-agent can assign it to pipeline agents.

The `name` field is the key used in pipeline configs (e.g. `"tools": ["my-server"]`). The entry point is taken from the first key in `[project.scripts]`. Each server gets its own `uv`-managed virtualenv on first run. The optional `timeout` field sets how long ADK waits for the server to become ready (default: 60s). On cold start `uv` installs dependencies into the server's virtualenv, which may take longer than usual.

### Transports

Auto-discovered servers always use **stdio** via `uv run`. When you pass `mcp_servers` as a dict, you can pick any supported transport.

Config helpers live in `fedotmas.mcp`:

```python
from fedotmas.mcp import (
    StdioMCPServer,   # any command + args over stdio
    HttpMCPServer,    # remote server over HTTP/SSE
    directory_server,  # uv run --directory
    workspace_server,  # uv run --package
    npx_server,        # npx -y <package>
    uvx_server,        # uvx --from <package>
    http_server,       # HTTP/SSE endpoint
)
```

#### HTTP/SSE (remote server)

```python
from fedotmas import MAS
from fedotmas.mcp import http_server

mas = MAS(mcp_servers={
    "my-remote-server": http_server(
        "https://mcp.example.com/sse",
        headers={"Authorization": "Bearer <token>"},
        description="Remote analytics server",
    ),
})
```

#### Docker (stdio)

Any MCP server that speaks stdio works with `docker run -i`. Here is an example with the [MongoDB MCP server](https://hub.docker.com/mcp/server/mongodb/overview):

```python
from fedotmas import MAS
from fedotmas.mcp import StdioMCPServer

mas = MAS(mcp_servers={
    "mongodb": StdioMCPServer(
        command="docker",
        args=(
            "run", "-i", "--rm",
            "-e", "MDB_MCP_CONNECTION_STRING",
            "mcp/mongodb",
        ),
        env={
            "MDB_MCP_CONNECTION_STRING": "mongodb+srv://user:pass@cluster.mongodb.net/mydb",
        },
        description="MongoDB database access via MCP",
        tags=("database",),
    ),
})
```

#### Mixing transports

You can combine auto-discovered, remote, and Docker servers in one registry:

```python
from fedotmas import MAS
from fedotmas.mcp import StdioMCPServer, http_server, discover_servers

# Start with auto-discovered workspace servers
servers = discover_servers()

# Add a remote server
servers["analytics"] = http_server(
    "https://mcp.example.com/sse",
    description="Remote analytics",
)

# Add a Docker server
servers["mongodb"] = StdioMCPServer(
    command="docker",
    args=("run", "-i", "--rm", "-e", "MDB_MCP_CONNECTION_STRING", "mcp/mongodb"),
    env={"MDB_MCP_CONNECTION_STRING": "mongodb+srv://..."},
    description="MongoDB access",
)

mas = MAS(mcp_servers=servers)
```
