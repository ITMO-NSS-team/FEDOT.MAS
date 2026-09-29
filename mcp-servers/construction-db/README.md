# construction-db MCP server

Read-only, bounded access to a restored STAIRS PostgreSQL database.

## Setup

Copy `.env.example` to a private environment file. Configure
`CONSTRUCTION_DB_STAIRS_DSN` for the STAIRS PostgreSQL instance. Use a database
role with only `CONNECT` and `SELECT` permissions; the server also begins every
operation in a read-only transaction.

Restore STAIRS with a compatible PostgreSQL toolchain.

```sh
cd mcp-servers/construction-db
uv sync
set -a; source .env; set +a
uv run mcp-construction-db
```

The server is discovered automatically as `construction-db`. After exporting
the variables from `.env.example`, verify it with:

```sh
just doctor construction-db
```

## Tools

| Tool | Purpose |
| --- | --- |
| `list_connections()` | List the `stairs`/PostgreSQL profile and configuration status. |
| `list_tables(connection)` | List non-system tables and views for `stairs`. |
| `describe_table(connection, table_name)` | Return columns for a `schema.table` or public table. |
| `query(connection, sql)` | Execute one bounded `SELECT` or `WITH ... SELECT` query for a profile. |
| `get_table_sample(connection, table_name, limit=20)` | Return a bounded sample for a profile. |

`query` rejects multiple statements and mutating/transaction/DDL keywords.
Every database operation uses a read-only transaction and the configured
`statement_timeout`; `CONSTRUCTION_DB_MAX_ROWS` is a hard cap on returned rows,
including schema-listing results. SQL filtering is defense in depth: use a
database role with `SELECT` permissions only.
