"""Read-only, reproducible audit of the restored SAMPO naming data.

Requires psycopg. Run with the construction-db environment:
  uv run --directory mcp-servers/construction-db python ../../scripts/audit_sampo.py
"""

from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path
from typing import Any

import psycopg
from psycopg import sql
from psycopg.rows import dict_row

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "sampo_audit"
TABLES = [
    "works_names_mv",
    "sampo_historical_data",
    "names_mapper",
    "resources_names_mv",
    "mschm_model",
    "perf_model",
    "res_model",
]
NAMING_FIELDS = {
    "works_names_mv": ["name", "processed_name", "granulary_name", "typed_lvl2_name"],
    "names_mapper": ["name", "processed_name", "granulary_name", "typed_lvl2_name"],
    "resources_names_mv": [
        "name",
        "processed_name",
        "granulary_name",
        "typed_lvl2_name",
    ],
    "sampo_historical_data": ["work_name", "granular_name"],
}


def dsn() -> str:
    return (
        os.getenv("SAMPO_AUDIT_DSN")
        or os.getenv("CONSTRUCTION_DB_SAMPO_DSN")
        or "dbname=sampo"
    )


def query(
    connection: psycopg.Connection,
    statement: sql.Composable,
    params: tuple[Any, ...] = (),
) -> list[dict[str, Any]]:
    with connection.cursor(row_factory=dict_row) as cursor:
        cursor.execute(statement, params)
        return list(cursor.fetchall())


def one(
    connection: psycopg.Connection,
    statement: sql.Composable,
    params: tuple[Any, ...] = (),
) -> dict[str, Any]:
    return query(connection, statement, params)[0]


def table_sql(table: str) -> sql.Identifier:
    return sql.Identifier("public", table)


def field_sql(field: str) -> sql.Identifier:
    return sql.Identifier(field)


def schema(connection: psycopg.Connection) -> dict[str, Any]:
    rows = query(
        connection,
        sql.SQL(
            "SELECT table_name, column_name, data_type, is_nullable, ordinal_position "
            "FROM information_schema.columns "
            "WHERE table_schema = 'public' AND table_name = ANY(%s) "
            "ORDER BY table_name, ordinal_position"
        ),
        (TABLES,),
    )
    result: dict[str, Any] = {table: [] for table in TABLES}
    for row in rows:
        result[row.pop("table_name")].append(row)
    return result


def naming_profiles(
    connection: psycopg.Connection, table: str, row_count: int
) -> dict[str, dict[str, Any]]:
    """Profile all naming fields in one source-table pass.

    The largest table has 3M+ rows. One multi-aggregate scan avoids repeating
    expensive full-table scans merely to profile adjacent columns.
    """
    fields = NAMING_FIELDS[table]
    source = table_sql(table)
    expressions: list[sql.Composable] = [sql.SQL("count(*) AS rows")]
    for field in fields:
        column = field_sql(field)
        expressions.append(
            sql.SQL("count(*) FILTER (WHERE {} IS NULL) AS {}").format(
                column, sql.Identifier(f"{field}__nulls")
            )
        )
        if row_count <= 200_000:
            expressions.append(
                sql.SQL("count(DISTINCT {}) AS {}").format(
                    column, sql.Identifier(f"{field}__distinct")
                )
            )
    aggregate = one(
        connection,
        sql.SQL("SELECT {} FROM {}").format(sql.SQL(", ").join(expressions), source),
    )
    result = {}
    estimates = {}
    if row_count > 200_000:
        stats_rows = query(
            connection,
            sql.SQL(
                "SELECT attname, n_distinct FROM pg_stats "
                "WHERE schemaname = 'public' AND tablename = %s AND attname = ANY(%s)"
            ),
            (table, fields),
        )
        estimates = {row["attname"]: row["n_distinct"] for row in stats_rows}
    for field in fields:
        nulls = aggregate[f"{field}__nulls"]
        distinct = aggregate.get(f"{field}__distinct")
        profile = {
            "rows": aggregate["rows"],
            "nulls": nulls,
        }
        if distinct is not None:
            profile["distinct_non_null"] = distinct
            profile["duplicate_rows"] = row_count - distinct - nulls
        else:
            estimate = estimates.get(field)
            if estimate is not None:
                estimate = (
                    round(abs(estimate) * row_count)
                    if estimate < 0
                    else round(estimate)
                )
            profile["distinct_non_null_estimate"] = estimate
            profile["distinct_estimation_method"] = "pg_stats.n_distinct"
        # Top-value scans add little value for 3M+ rows and can dominate audit
        # runtime. The large-table cardinality estimate is explicitly labelled;
        # detailed top values are collected from smaller label tables.
        if row_count <= 200_000:
            profile["top_values"] = query(
                connection,
                sql.SQL(
                    "SELECT {column} AS value, count(*) AS frequency FROM {source} "
                    "WHERE {column} IS NOT NULL GROUP BY {column} ORDER BY frequency DESC, value LIMIT 10"
                ).format(source=source, column=field_sql(field)),
            )
        else:
            profile["top_values"] = []
            profile["top_values_note"] = (
                "Skipped to avoid an additional full-table grouping scan."
            )
        result[field] = profile
    return result


def relation_candidate(
    connection: psycopg.Connection,
    source_table: str,
    source_name: str,
    source_key: str,
    target_table: str,
    target_key: str,
    target_name: str,
) -> dict[str, Any]:
    """Measure a mapping through observed equality of source_key and target_key."""
    statement = sql.SQL(
        "WITH source_values AS ("
        "  SELECT {source_key} AS join_key, {source_name} AS source_name, count(*) AS source_frequency "
        "  FROM {source_table} WHERE {source_key} IS NOT NULL AND {source_name} IS NOT NULL "
        "  GROUP BY 1, 2"
        "), target_values AS ("
        "  SELECT {target_key} AS join_key, {target_name} AS target_name "
        "  FROM {target_table} WHERE {target_key} IS NOT NULL AND {target_name} IS NOT NULL GROUP BY 1, 2"
        "), pairs AS ("
        "  SELECT s.source_name, t.target_name, s.source_frequency "
        "  FROM source_values s JOIN target_values t USING (join_key)"
        "), source_cardinality AS ("
        "  SELECT source_name, count(DISTINCT target_name) AS labels FROM pairs GROUP BY 1"
        "), target_cardinality AS ("
        "  SELECT target_name, count(DISTINCT source_name) AS labels FROM pairs GROUP BY 1"
        ") "
        "SELECT "
        " (SELECT coalesce(sum(source_frequency), 0) FROM source_values s WHERE EXISTS (SELECT 1 FROM target_values t WHERE t.join_key = s.join_key)) AS matched_rows, "
        " (SELECT count(*) FROM source_cardinality) AS unique_source_names, "
        " (SELECT count(*) FROM target_cardinality) AS unique_target_names, "
        " (SELECT count(*) FROM source_cardinality WHERE labels > 1) AS one_to_many_source_names, "
        " (SELECT count(*) FROM target_cardinality WHERE labels > 1) AS many_to_one_target_names, "
        " (SELECT count(*) FROM source_cardinality WHERE labels > 1) AS conflicting_labels, "
        " (SELECT count(*) FROM {source_table} WHERE {source_key} IS NULL OR {source_name} IS NULL) AS source_null_rows, "
        " (SELECT count(*) FROM {target_table} WHERE {target_key} IS NULL OR {target_name} IS NULL) AS target_null_rows"
    ).format(
        source_table=table_sql(source_table),
        source_name=field_sql(source_name),
        source_key=field_sql(source_key),
        target_table=table_sql(target_table),
        target_key=field_sql(target_key),
        target_name=field_sql(target_name),
    )
    return {
        "source": f"{source_table}.{source_name}",
        "target": f"{target_table}.{target_name}",
        "join": f"{source_table}.{source_key} = {target_table}.{target_key}",
        **one(connection, statement),
    }


def direct_historical_mapping(
    connection: psycopg.Connection,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pairs = query(
        connection,
        sql.SQL(
            "SELECT work_name AS source_work_name, granular_name AS target_granular_name, count(*) AS row_frequency "
            "FROM public.sampo_historical_data WHERE work_name IS NOT NULL AND granular_name IS NOT NULL "
            "GROUP BY 1, 2"
        ),
    )
    by_source: dict[str, list[dict[str, Any]]] = {}
    for pair in pairs:
        by_source.setdefault(pair["source_work_name"], []).append(pair)
    decisions = [
        {
            "source_work_name": source,
            "target_granular_name": values[0]["target_granular_name"],
            "source_row_count": sum(value["row_frequency"] for value in values),
        }
        for source, values in by_source.items()
        if len({value["target_granular_name"] for value in values}) == 1
    ]
    ambiguous = len(by_source) - len(decisions)
    normalized = lambda value: re.sub(r"[\W_]+", "", value.casefold(), flags=re.UNICODE)
    exact = sum(d["source_work_name"] == d["target_granular_name"] for d in decisions)
    normalized_exact = sum(
        normalized(d["source_work_name"]) == normalized(d["target_granular_name"])
        for d in decisions
    )
    frequencies = sorted(d["source_row_count"] for d in decisions)
    median = frequencies[len(frequencies) // 2] if frequencies else 0
    easy = next(
        (
            d
            for d in decisions
            if normalized(d["source_work_name"])
            == normalized(d["target_granular_name"])
        ),
        None,
    )
    medium = next(
        (d for d in decisions if d != easy and d["source_row_count"] >= median), None
    )
    hard = next(
        (
            d
            for d in sorted(
                decisions, key=lambda item: item["source_row_count"], reverse=True
            )
            if normalized(d["source_work_name"])
            != normalized(d["target_granular_name"])
        ),
        None,
    )
    stats = {
        "formulation": "historical work_name -> observed historical granular_name",
        "evidence": "Both values occur in the same historical record; this is an observed label relationship, not a semantic claim based only on names.",
        "source_rows_with_non_null_pair": sum(pair["row_frequency"] for pair in pairs),
        "unique_source_names": len(by_source),
        "clean_unique_mapping_decisions": len(decisions),
        "ambiguous_source_names": ambiguous,
        "conflicting_labels": ambiguous,
        "unique_target_names": len({d["target_granular_name"] for d in decisions}),
        "exact_match_baseline_accuracy": exact / len(decisions) if decisions else None,
        "normalized_exact_match_baseline_accuracy": normalized_exact / len(decisions)
        if decisions
        else None,
        "median_source_frequency": median,
        "examples": {"easy": easy, "medium": medium, "hard": hard},
    }
    return stats, sorted(decisions, key=lambda item: item["source_work_name"])


def write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (
        psycopg.connect(
            dsn(), options="-c default_transaction_read_only=on"
        ) as connection,
        connection.transaction(),
    ):
        with connection.cursor() as cursor:
            cursor.execute("SET TRANSACTION READ ONLY")
            cursor.execute("SET LOCAL statement_timeout = 600000")
        schemas = schema(connection)
        row_counts = {
            table: one(
                connection,
                sql.SQL("SELECT count(*) AS rows FROM {}").format(table_sql(table)),
            )["rows"]
            for table in TABLES
        }
        profiles = {
            table: naming_profiles(connection, table, row_counts[table])
            for table in NAMING_FIELDS
        }
        # Test the two observed routes that could support the proposed task:
        # raw historical name directly matching a mapper name, and its observed
        # historical granular label matching the mapper granular label.  The
        # full 2x4 cross-product is redundant and needlessly expensive.
        historical_to_mapper = [
            relation_candidate(
                connection,
                "sampo_historical_data",
                "work_name",
                history_key,
                "names_mapper",
                mapper_key,
                target_name,
            )
            for history_key, mapper_key, target_name in (
                ("work_name", "name", "granulary_name"),
                ("granular_name", "granulary_name", "name"),
            )
        ]
        mapper_to_works = [
            relation_candidate(
                connection,
                "names_mapper",
                "name",
                "name",
                "works_names_mv",
                "name",
                "granulary_name",
            )
        ]
        task, ground_truth = direct_historical_mapping(connection)
    write_json(OUTPUT / "schema.json", {"tables": schemas, "row_counts": row_counts})
    write_json(
        OUTPUT / "stats.json",
        {
            "row_counts": row_counts,
            "naming_field_profiles": profiles,
            "historical_to_mapper_candidates": historical_to_mapper,
            "mapper_to_works_candidates": mapper_to_works,
            "selected_task": task,
            "leakage": {
                "must_hide": [
                    "sampo_historical_data.granular_name",
                    "names_mapper",
                    "works_names_mv",
                ],
                "reason": "The first is the observed target label; the latter tables can expose exact joins or aliases used to derive labels.",
            },
        },
    )
    with (OUTPUT / "private_ground_truth.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["source_work_name", "target_granular_name", "source_row_count"],
        )
        writer.writeheader()
        writer.writerows(ground_truth)
    report = [
        "# SAMPO work-name entity-resolution audit",
        "",
        "This report is generated from read-only database queries; it does not infer labels solely from column names.",
        "",
        "## Row counts",
        "",
        *[f"- `{table}`: {count:,}" for table, count in row_counts.items()],
        "",
        "## Selected formulation",
        "",
        f"`{task['formulation']}`",
        "",
        task["evidence"],
        "",
        f"- Clean decisions: {task['clean_unique_mapping_decisions']:,} / {task['unique_source_names']:,} unique sources",
        f"- Ambiguous sources / conflicting labels: {task['ambiguous_source_names']:,}",
        f"- Exact baseline: {task['exact_match_baseline_accuracy']:.4f}",
        f"- Normalized exact baseline: {task['normalized_exact_match_baseline_accuracy']:.4f}",
        "",
        "## Leakage",
        "",
        "Hide `sampo_historical_data.granular_name`, all of `names_mapper`, and all of `works_names_mv` from evaluated systems. They contain labels or joinable aliases. Detailed candidate-join metrics are in `stats.json`.",
    ]
    (OUTPUT / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote SAMPO audit to {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
