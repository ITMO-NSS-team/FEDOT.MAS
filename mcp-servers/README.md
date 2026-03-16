## MCP servers

FEDOT.MAS uses [MCP](https://modelcontextprotocol.io/) (Model Context Protocol) to give agents access to external tools like file downloads, web scraping, and code execution.

When a user describes a task, the **meta-agent** reads the descriptions of all registered MCP servers and decides which tools each pipeline agent needs. At runtime the pipeline builder wraps each server in an ADK `McpToolset`. The ADK then either launches a subprocess (for stdio servers) or opens an HTTP connection (for remote ones).

### Auto-discovery

Discovery runs when you pass `mcp_servers="all"` or a list of server names (e.g. `["download-url-content"]`). The registry walks up from the `fedotmas` package to find the workspace root (a `pyproject.toml` with `[tool.uv.workspace]`), then scans every `mcp-servers/*/pyproject.toml` for `[tool.fedotmas.mcp]` sections.

If you leave `mcp_servers` at the default `None`, no discovery happens and no servers are registered.

If `fedotmas` is installed from PyPI (outside a workspace), discovery finds no root and returns an empty registry. You can still use servers by passing a config dict directly or by pointing `discover_local_servers()` at a directory with the same layout:

```python
from fedotmas import MAW, discover_local_servers

maw = MAW(mcp_servers=discover_local_servers("/path/to/my-servers"))
```

After construction you can inspect the resolved registry:

```python
print(maw.mcp_servers)  # {'light-sandbox': StdioMCPServer(...), ...}
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

2. Add a `[tool.fedotmas.mcp]` section to `pyproject.toml`:

```toml
[tool.fedotmas.mcp]
name = "my-server"
description = "Short description of what the server does. The meta-agent reads this to decide when to use it."
tags = ["relevant", "tags"]
timeout = 120
```

For Python servers, also add `[project]` + `[project.scripts]` so the discovery knows which entry point to launch via `uv run`:

```toml
[project]
name = "my-mcp-server"
version = "0.1.0"
dependencies = ["fastmcp>=2.14.5"]

[project.scripts]
my-mcp-server = "my_server.server:main"
```

For external binaries that already speak MCP stdio, use `command` + `args` instead of `[project.scripts]`:

```toml
[tool.fedotmas.mcp]
name = "browser"
command = "lightpanda"
args = ["mcp"]
```

3. Next time you call `MAS(mcp_servers="all")`, the new server appears in the registry and the meta-agent can assign it to pipeline agents.

### `[tool.fedotmas.mcp]` fields

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `name` | yes | — | Registry key used in pipeline configs (`"tools": ["my-server"]`) |
| `description` | no | `""` | Human-readable description. The meta-agent reads this to decide when to assign the server. |
| `tags` | no | `[]` | List of tags for categorization and filtering. |
| `timeout` | no | `60` | Seconds to wait for MCP session to become ready. |
| `command` | no | — | Path or name of an external binary that speaks MCP stdio. When set, `[project.scripts]` is not required. |
| `args` | no | `[]` | Arguments passed to `command`. Only used when `command` is set. |

**Resolution order:** if `command` is present, discovery creates a `StdioMCPServer` pointing directly at that binary. Otherwise it reads the first key from `[project.scripts]` and launches via `uv run --directory`. Python servers get their own `uv`-managed virtualenv on first run (cold start installs dependencies, which may take longer than usual).

### Transports

Auto-discovered servers always use **stdio** via `uv run`. When you pass `mcp_servers` as a dict, you can use either transport directly:

- `StdioMCPServer` is any command + args over stdio
- `HttpMCPServer` is remote server over HTTP (Streamable HTTP transport)

#### HTTP (remote server)

```python
from fedotmas import MAW, HttpMCPServer

maw = MAW(mcp_servers={
    "my-remote-server": HttpMCPServer(
        url="https://mcp.example.com",
        headers={"Authorization": "Bearer <token>"},
        description="Remote analytics server",
    ),
})
```

#### Docker (stdio)

Any MCP server that speaks stdio works with `docker run -i`. Here is an example with the [MongoDB MCP server](https://hub.docker.com/mcp/server/mongodb/overview):

```python
from fedotmas import MAW, StdioMCPServer

maw = MAW(mcp_servers={
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
from fedotmas import MAW, StdioMCPServer, HttpMCPServer, discover_local_servers

# Start with auto-discovered workspace servers
servers = discover_local_servers()

# Add a remote server
servers["analytics"] = HttpMCPServer(
    url="https://mcp.example.com",
    description="Remote analytics",
)

# Add a Docker server
servers["mongodb"] = StdioMCPServer(
    command="docker",
    args=("run", "-i", "--rm", "-e", "MDB_MCP_CONNECTION_STRING", "mcp/mongodb"),
    env={"MDB_MCP_CONNECTION_STRING": "mongodb+srv://..."},
    description="MongoDB access",
)

maw = MAW(mcp_servers=servers)
```
