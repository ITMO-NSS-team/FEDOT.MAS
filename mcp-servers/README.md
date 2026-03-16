# searxng-search MCP server

FEDOT.MAS uses [MCP](https://modelcontextprotocol.io/) (Model Context Protocol) to give agents access to external tools like file downloads, web scraping, and code execution.

When a user describes a task, the **meta-agent** reads the descriptions of all registered MCP servers and decides which tools each pipeline agent needs. At runtime the pipeline builder wraps each server in an ADK `McpToolset`. The ADK then either launches a subprocess (for stdio servers) or opens an HTTP connection (for remote ones).

### Set up SearXNG

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

**Prerequisites:** Docker and Docker Compose installed

**Manual setup (without just):**

```bash
# Clone SearXNG Docker repository
cd ~
git clone https://github.com/searxng/searxng-docker.git
cd searxng-docker

# Generate secret key
sed -i "s|ultrasecretkey|$(openssl rand -hex 32)|g" searxng/settings.yml

# Configure search formats
cat >> searxng/settings.yml << EOF
search:
  formats:
    - html
    - json
    - csv
    - rss
EOF

# Update port in docker-compose.yaml
sed -i "s/127.0.0.1:8080:8080/127.0.0.1:8888:8080/" docker-compose.yaml

# Start services
docker compose up -d

# Verify running
curl http://localhost:8888
```

2. Add the standard project metadata **and** a `[tool.fedotmas.mcp]` section to `pyproject.toml`:


## Usage in FEDOT.MAS

[tool.fedotmas.mcp]
name = "my-server"
description = "Short description of what the server does. The meta-agent reads this to decide when to use it."
tags = ["relevant", "tags"]
timeout = 120  # optional, seconds to wait for MCP session (default: 60)
```

3. Next time you call `MAS(mcp_servers="all")`, the new server appears in the registry and the meta-agent can assign it to pipeline agents.

The `name` field is the key used in pipeline configs (e.g. `"tools": ["my-server"]`). The entry point is taken from the first key in `[project.scripts]`. Each server gets its own `uv`-managed virtualenv on first run. The optional `timeout` field sets how long ADK waits for the server to become ready (default: 60s). On cold start `uv` installs dependencies into the server's virtualenv, which may take longer than usual.

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
