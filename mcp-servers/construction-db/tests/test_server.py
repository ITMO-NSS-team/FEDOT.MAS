import pytest

from mcp_construction_db.server import (
    QueryRejected,
    bounded_query,
    connection_string,
    validate_select_sql,
)


def test_read_only_enforcement_rejects_mutating_and_multi_statement_sql() -> None:
    for sql in (
        "DELETE FROM works",
        "SELECT * FROM works; DELETE FROM works",
        "WITH removed AS (DELETE FROM works RETURNING *) SELECT * FROM removed",
        "SELECT 1; SELECT 2",
        "SELECT id INTO copied_works FROM works",
        "SELECT * FROM works FOR SHARE",
        "WITH rows AS (SELECT 1) TABLE rows",
        "SELECT set_config('statement_timeout', '0', false)",
    ):
        try:
            validate_select_sql(sql)
        except QueryRejected:
            pass
        else:
            raise AssertionError(f"Expected SQL to be rejected: {sql}")


def test_bounded_query_wraps_select_with_parameterized_limit() -> None:
    sql, parameters = bounded_query("SELECT id FROM public.works", 25)

    assert sql == "SELECT * FROM (SELECT id FROM public.works) AS construction_db_result LIMIT %s"
    assert parameters == (25,)


def test_validator_accepts_literals_and_a_top_level_with_select() -> None:
    assert validate_select_sql("SELECT 'DELETE; UPDATE' AS text") == "SELECT 'DELETE; UPDATE' AS text"
    assert validate_select_sql("WITH item AS (SELECT 1) SELECT * FROM item") == "WITH item AS (SELECT 1) SELECT * FROM item"
    assert validate_select_sql("SELECT $$DELETE; UPDATE$$ AS text") == "SELECT $$DELETE; UPDATE$$ AS text"


def test_connection_profiles_are_explicit_and_separate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONSTRUCTION_DB_STAIRS_DSN", "postgresql://stairs")
    monkeypatch.setenv("CONSTRUCTION_DB_SAMPO_DSN", "postgresql://sampo")

    assert connection_string("stairs") == "postgresql://stairs"
    assert connection_string("sampo") == "postgresql://sampo"
    with pytest.raises(RuntimeError, match="Unknown connection"):
        connection_string("combined")
