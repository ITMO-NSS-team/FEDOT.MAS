from __future__ import annotations

import datetime as dt
import decimal
import os
import re
from collections.abc import Iterator
from typing import Any

import psycopg
from fastmcp import FastMCP
from psycopg.rows import dict_row

mcp = FastMCP("construction-db")

_CONNECTION_PROFILES = {
    "stairs": {
        "dsn_env": "CONSTRUCTION_DB_STAIRS_DSN",
        "engine": "PostgreSQL",
        "source": "STAIRS",
    },
    "sampo": {
        "dsn_env": "CONSTRUCTION_DB_SAMPO_DSN",
        "engine": "Greenplum",
        "source": "SAMPO",
    },
}

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")
_MUTATING_KEYWORDS = {
    "alter",
    "analyze",
    "begin",
    "call",
    "checkpoint",
    "close",
    "cluster",
    "comment",
    "commit",
    "copy",
    "create",
    "deallocate",
    "delete",
    "discard",
    "do",
    "drop",
    "end",
    "execute",
    "grant",
    "insert",
    "listen",
    "load",
    "lock",
    "merge",
    "move",
    "notify",
    "prepare",
    "reassign",
    "refresh",
    "reindex",
    "release",
    "reset",
    "revoke",
    "rollback",
    "savepoint",
    "security",
    "set",
    "show",
    "truncate",
    "unlisten",
    "update",
    "vacuum",
    "for",
    "into",
}
_SIDE_EFFECT_FUNCTIONS = {
    "dblink_connect",
    "dblink_exec",
    "lo_export",
    "lo_import",
    "pg_advisory_lock",
    "pg_advisory_xact_lock",
    "pg_cancel_backend",
    "pg_reload_conf",
    "pg_terminate_backend",
    "set_config",
}
_DOLLAR_QUOTE = re.compile(r"\$(?:[A-Za-z_][A-Za-z0-9_]*)?\$")


class QueryRejected(ValueError):
    """Raised when SQL is not exactly one read-only query."""


def _positive_int(name: str, default: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return min(max(value, 1), maximum)


def max_rows() -> int:
    return _positive_int("CONSTRUCTION_DB_MAX_ROWS", 100, 10_000)


def statement_timeout_ms() -> int:
    return _positive_int("CONSTRUCTION_DB_STATEMENT_TIMEOUT_MS", 5_000, 120_000)


def connection_string(connection: str) -> str:
    """Return a configured DSN without ever returning it through MCP."""
    profile = _CONNECTION_PROFILES.get(connection)
    if profile is None:
        available = ", ".join(sorted(_CONNECTION_PROFILES))
        raise RuntimeError(
            f"Unknown connection '{connection}'. Available connections: {available}."
        )
    if dsn := os.getenv(profile["dsn_env"]):
        return dsn
    raise RuntimeError(
        f"Connection '{connection}' is not configured. Set {profile['dsn_env']} to a read-only {profile['engine']} DSN."
    )


def _tokens(sql: str) -> Iterator[str]:
    """Yield SQL words and structural tokens while skipping literals/comments."""
    i, length = 0, len(sql)
    while i < length:
        char = sql[i]
        if char.isspace():
            i += 1
        elif sql.startswith("--", i):
            newline = sql.find("\n", i + 2)
            i = length if newline == -1 else newline + 1
        elif sql.startswith("/*", i):
            depth = 1
            i += 2
            while i < length and depth:
                if sql.startswith("/*", i):
                    depth += 1
                    i += 2
                elif sql.startswith("*/", i):
                    depth -= 1
                    i += 2
                else:
                    i += 1
            if depth:
                raise QueryRejected("Invalid SQL: unterminated block comment.")
        elif char == "'":
            i += 1
            while i < length:
                if sql[i] == "'":
                    if i + 1 < length and sql[i + 1] == "'":
                        i += 2
                        continue
                    i += 1
                    break
                else:
                    i += 1
            else:
                raise QueryRejected("Invalid SQL: unterminated string literal.")
        elif char == '"':
            i += 1
            while i < length:
                if sql[i] == '"':
                    if i + 1 < length and sql[i + 1] == '"':
                        i += 2
                        continue
                    i += 1
                    break
                else:
                    i += 1
            else:
                raise QueryRejected("Invalid SQL: unterminated quoted identifier.")
        elif char == ";":
            yield ";"
            i += 1
        elif char in "()":
            yield char
            i += 1
        elif char == "$" and (match := _DOLLAR_QUOTE.match(sql, i)):
            delimiter = match.group(0)
            end = sql.find(delimiter, match.end())
            if end == -1:
                raise QueryRejected("Invalid SQL: unterminated dollar-quoted literal.")
            i = end + len(delimiter)
        elif char.isalpha() or char == "_":
            start = i
            i += 1
            while i < length and (sql[i].isalnum() or sql[i] in "_$"):
                i += 1
            yield sql[start:i].casefold()
        else:
            i += 1


def validate_select_sql(sql: str) -> str:
    """Accept exactly one SELECT or WITH ... SELECT statement.

    This lexical gate is defense in depth. The database transaction is also
    explicitly read-only, so validation failure cannot become a write path.
    """
    tokens = list(_tokens(sql))
    if not tokens:
        raise QueryRejected("SQL is empty. Provide one SELECT statement.")
    semicolons = [index for index, token in enumerate(tokens) if token == ";"]
    if semicolons and semicolons != [len(tokens) - 1]:
        raise QueryRejected(
            "Only one SELECT statement is allowed; multi-statement SQL is rejected."
        )
    words = [token for token in tokens if token not in {";", "(", ")"}]
    if not words or words[0] not in {"select", "with"}:
        raise QueryRejected(
            "Only SELECT statements (including WITH ... SELECT) are allowed."
        )
    forbidden = sorted(set(words) & _MUTATING_KEYWORDS)
    if forbidden:
        raise QueryRejected(
            f"Only read-only SELECT is allowed; forbidden keyword: {forbidden[0].upper()}."
        )
    side_effects = sorted(set(words) & _SIDE_EFFECT_FUNCTIONS)
    if side_effects:
        raise QueryRejected(f"Query uses prohibited function: {side_effects[0]}().")
    if words[0] == "with":
        depth = 0
        top_level_words = []
        for token in tokens:
            if token == "(":
                depth += 1
            elif token == ")":
                depth -= 1
            elif token not in {";"} and depth == 0:
                top_level_words.append(token)
        if "select" not in top_level_words:
            raise QueryRejected("WITH queries must have a top-level SELECT result.")
    return sql.strip().rstrip(";").strip()


def bounded_query(sql: str, limit: int) -> tuple[str, tuple[int]]:
    return (
        f"SELECT * FROM ({validate_select_sql(sql)}) AS construction_db_result LIMIT %s",
        (limit,),
    )


def _table_parts(table_name: str) -> tuple[str, str]:
    parts = table_name.split(".")
    if len(parts) == 1:
        parts.insert(0, "public")
    if len(parts) != 2 or not all(_IDENTIFIER.fullmatch(part) for part in parts):
        raise ValueError(
            "Invalid table name. Use an unquoted name such as public.works or works."
        )
    return parts[0], parts[1]


def _quote_identifier(identifier: str) -> str:
    return f'"{identifier}"'


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (dt.date, dt.datetime, dt.time)):
        return value.isoformat()
    if isinstance(value, decimal.Decimal):
        return str(value)
    if isinstance(value, bytes):
        return f"<bytes:{len(value)}>"
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _run(
    connection_name: str, sql: str, parameters: tuple[Any, ...] = ()
) -> tuple[list[str], list[dict[str, Any]]]:
    try:
        with (
            psycopg.connect(
                connection_string(connection_name),
                options="-c default_transaction_read_only=on",
                row_factory=dict_row,
            ) as database_connection,
            database_connection.transaction(),
            database_connection.cursor() as cursor,
        ):
            cursor.execute("SET TRANSACTION READ ONLY")
            cursor.execute(f"SET LOCAL statement_timeout = {statement_timeout_ms()}")
            cursor.execute(sql, parameters)
            columns = (
                [column.name for column in cursor.description]
                if cursor.description
                else []
            )
            return columns, [_json_safe(row) for row in cursor.fetchall()]
    except psycopg.Error as error:
        raise RuntimeError(_error_message(error)) from None


def _error_message(error: Exception) -> str:
    sqlstate = getattr(error, "sqlstate", None)
    if sqlstate == "42P01":
        return "Table or relation does not exist. Call list_tables() to find available names."
    if sqlstate == "57014":
        return f"Query timed out after {statement_timeout_ms()} ms. Simplify the query or add a selective WHERE clause."
    if sqlstate in {"42601", "42703"}:
        return f"Invalid SQL: {getattr(error, 'diag', None).message_primary if getattr(error, 'diag', None) else str(error)}"
    return f"Database query failed: {str(error)[:300]}"


@mcp.tool
def list_connections() -> dict[str, Any]:
    """List STAIRS/PostgreSQL and SAMPO/Greenplum connection profiles without credentials."""
    return {
        "connections": [
            {
                "name": name,
                "source": profile["source"],
                "engine": profile["engine"],
                "configured": bool(os.getenv(profile["dsn_env"])),
            }
            for name, profile in sorted(_CONNECTION_PROFILES.items())
        ]
    }


@mcp.tool
def list_tables(connection: str) -> dict[str, Any]:
    """List accessible non-system tables and views for stairs or sampo."""
    try:
        columns, rows = _run(
            connection,
            "SELECT table_schema AS schema, table_name AS name, table_type "
            "FROM information_schema.tables "
            "WHERE table_schema NOT IN ('pg_catalog', 'information_schema') "
            "ORDER BY table_schema, table_name LIMIT %s",
            (max_rows(),),
        )
    except RuntimeError as error:
        return {"error": str(error)}
    return {
        "columns": columns,
        "rows": rows,
        "row_count": len(rows),
        "row_limit": max_rows(),
        "truncated": len(rows) == max_rows(),
    }


@mcp.tool
def describe_table(connection: str, table_name: str) -> dict[str, Any]:
    """Describe a table in the stairs or sampo connection."""
    try:
        schema, table = _table_parts(table_name)
    except ValueError as error:
        return {"error": str(error)}
    try:
        columns, rows = _run(
            connection,
            "SELECT column_name, data_type, is_nullable, column_default, ordinal_position "
            "FROM information_schema.columns WHERE table_schema = %s AND table_name = %s "
            "ORDER BY ordinal_position LIMIT %s",
            (schema, table, max_rows()),
        )
    except RuntimeError as error:
        return {"error": str(error)}
    if not rows:
        return {
            "error": f"Table {schema}.{table} does not exist or is not accessible. Call list_tables() first."
        }
    return {"table": f"{schema}.{table}", "columns": columns, "rows": rows}


@mcp.tool
def query(connection: str, sql: str) -> dict[str, Any]:
    """Run one bounded read-only SELECT query on stairs or sampo."""
    try:
        statement, parameters = bounded_query(sql, max_rows())
    except QueryRejected as error:
        return {"error": str(error)}
    try:
        columns, rows = _run(connection, statement, parameters)
    except RuntimeError as error:
        return {"error": str(error)}
    return {
        "columns": columns,
        "rows": rows,
        "row_count": len(rows),
        "row_limit": max_rows(),
        "truncated": len(rows) == max_rows(),
    }


@mcp.tool
def get_table_sample(
    connection: str, table_name: str, limit: int = 20
) -> dict[str, Any]:
    """Return a bounded table sample from the stairs or sampo connection."""
    try:
        schema, table = _table_parts(table_name)
    except ValueError as error:
        return {"error": str(error)}
    requested = max(1, min(limit, max_rows()))
    statement = (
        f"SELECT * FROM {_quote_identifier(schema)}.{_quote_identifier(table)} LIMIT %s"
    )
    try:
        columns, rows = _run(connection, statement, (requested,))
    except RuntimeError as error:
        return {"error": str(error)}
    return {
        "table": f"{schema}.{table}",
        "columns": columns,
        "rows": rows,
        "row_count": len(rows),
        "row_limit": requested,
        "truncated": len(rows) == requested,
    }


def main() -> None:
    mcp.run(show_banner=False)
