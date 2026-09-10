#!/usr/bin/env python3
"""Read-only audit of technological-card source files and parsed Markdown."""

from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
import re
import shutil
import subprocess
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any


CARD_EXTENSIONS = {".doc", ".docx", ".pdf", ".rtf"}
DUMP_EXTENSIONS = {".sql", ".dump", ".backup", ".tar"}


def normalize_filename(value: str) -> str:
    """Normalize only Unicode form, case, and separator whitespace.

    Punctuation and words are deliberately retained so matching does not hide
    filename changes.  Callers should pass a filename stem or directory name.
    """
    value = unicodedata.normalize("NFKC", value).casefold()
    value = re.sub(r"[\s_]+", " ", value)
    return value.strip()


def card_key(category: str, title: str) -> tuple[str, str]:
    return normalize_filename(category), normalize_filename(title)


def card_id(key: tuple[str, str]) -> str:
    return hashlib.sha256("\0".join(key).encode("utf-8")).hexdigest()[:16]


def relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def collect_cards(root: Path, kind: str, repo_root: Path) -> dict[tuple[str, str], list[dict[str, Any]]]:
    records: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    allowed = {".md"} if kind == "markdown" else CARD_EXTENSIONS
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.casefold() not in allowed:
            continue
        category = path.relative_to(root).parts[0] if len(path.relative_to(root).parts) > 1 else ""
        title = normalize_filename(path.stem)
        item: dict[str, Any] = {
            "path": relative(path, repo_root),
            "category": category,
            "normalized_title": title,
        }
        if kind == "raw":
            item["raw_extension"] = path.suffix.casefold()
        else:
            text = path.read_text(encoding="utf-8", errors="replace")
            item["bytes"] = path.stat().st_size
            item["non_whitespace_characters"] = len(text.strip())
        records[card_key(category, path.stem)].append(item)
    return records


def build_manifest(
    markdown: dict[tuple[str, str], list[dict[str, Any]]],
    raw: dict[tuple[str, str], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    manifest = []
    for key in sorted(set(markdown) | set(raw)):
        markdown_records, raw_records = markdown.get(key, []), raw.get(key, [])
        md_paths = [item["path"] for item in markdown_records]
        raw_paths = [item["path"] for item in raw_records]
        manifest.append(
            {
                "card_id": card_id(key),
                "category": (markdown_records or raw_records)[0]["category"],
                "markdown_path": md_paths[0] if len(md_paths) == 1 else None,
                "raw_path": raw_paths[0] if len(raw_paths) == 1 else None,
                "raw_extension": raw_records[0]["raw_extension"] if len(raw_records) == 1 else None,
                "normalized_title": key[1],
                "has_markdown": bool(markdown_records),
                "has_raw": bool(raw_records),
                "markdown_paths": md_paths,
                "raw_paths": raw_paths,
            }
        )
    return manifest


def filename_candidates(records: dict[tuple[str, str], list[dict[str, Any]]], source: str) -> list[dict[str, Any]]:
    """Report conservative fuzzy candidates within a category, never matches."""
    by_category: dict[str, list[tuple[str, list[dict[str, Any]]]]] = defaultdict(list)
    for (category, title), paths in records.items():
        by_category[category].append((title, paths))
    candidates = []
    for category, entries in sorted(by_category.items()):
        for index, (left, left_paths) in enumerate(entries):
            for right, right_paths in entries[index + 1 :]:
                if left == right or left[:8] != right[:8]:
                    continue
                ratio = difflib.SequenceMatcher(None, left, right).ratio()
                if ratio >= 0.97:
                    candidates.append(
                        {
                            "source": source,
                            "category": category,
                            "left_title": left,
                            "right_title": right,
                            "similarity": round(ratio, 4),
                            "left_paths": [item["path"] for item in left_paths],
                            "right_paths": [item["path"] for item in right_paths],
                        }
                    )
    return candidates


def parse_plain_postgres(text: str) -> tuple[list[str], list[dict[str, str]]]:
    schemas = sorted(set(re.findall(r"(?im)^CREATE SCHEMA(?: IF NOT EXISTS)?\s+(?:\"([^\"]+)\"|([A-Za-z_][\w$]*))", text)))
    schema_names = sorted({quoted or bare for quoted, bare in schemas})
    tables = []
    pattern = re.compile(
        r"(?im)^CREATE(?: UNLOGGED)? TABLE(?: IF NOT EXISTS)?\s+"
        r"(?:(\"[^\"]+\"|[A-Za-z_][\w$]*)\.)?(\"[^\"]+\"|[A-Za-z_][\w$]*)"
    )
    for schema, table in pattern.findall(text):
        tables.append({"schema": schema.strip('"') if schema else None, "table": table.strip('"')})
    return schema_names, tables


def inspect_dump(path: Path, repo_root: Path) -> dict[str, Any]:
    data = path.read_bytes()
    custom = data.startswith(b"PGDMP")
    result: dict[str, Any] = {
        "path": relative(path, repo_root),
        "size_bytes": len(data),
        "dump_type": "postgresql_custom_archive" if custom else "plain_sql" if path.suffix.casefold() == ".sql" or b"PostgreSQL database dump" in data[:4096] else "unrecognized_database_dump",
        "likely_db_engine": "PostgreSQL",
        "engine_evidence": [],
        "schemas": [],
        "tables": [],
        "appears_restorable": False,
        "restore_assessment": "not assessed",
    }
    if b"gpadmin" in data or "greenplum" in path.name.casefold() or "_gp_" in path.name.casefold():
        result["likely_db_engine"] = "PostgreSQL-compatible; Greenplum likely"
        result["engine_evidence"].append("Greenplum naming or gpadmin metadata")
    if custom:
        pg_restore = shutil.which("pg_restore")
        if not pg_restore:
            result["restore_assessment"] = "pg_restore unavailable; archive magic confirms custom PostgreSQL format only"
            return result
        completed = subprocess.run([pg_restore, "--list", str(path)], capture_output=True, text=True, check=False)
        if completed.returncode:
            result["restore_assessment"] = f"pg_restore --list failed: {completed.stderr.strip()[:300]}"
            return result
        result["appears_restorable"] = True
        result["restore_assessment"] = "pg_restore --list succeeded; restore still requires a compatible server and roles"
        for line in completed.stdout.splitlines():
            match = re.search(r"\bSCHEMA\s+-\s+(\S+)", line)
            if match:
                result["schemas"].append(match.group(1))
            match = re.search(r"\bTABLE\s+(\S+)\s+(\S+)", line)
            if match and "TABLE DATA" not in line:
                result["tables"].append({"schema": match.group(1), "table": match.group(2)})
        result["schemas"] = sorted(set(result["schemas"]))
        result["tables"] = sorted(result["tables"], key=lambda item: (item["schema"], item["table"]))
        return result
    text = data.decode("utf-8", errors="replace")
    schemas, tables = parse_plain_postgres(text)
    result["schemas"], result["tables"] = schemas, tables
    if "PostgreSQL database dump" in text[:4096]:
        result["engine_evidence"].append("PostgreSQL dump header")
        result["appears_restorable"] = bool(tables or schemas)
        result["restore_assessment"] = "plain pg_dump SQL contains schema objects; restore with psql against a compatible PostgreSQL server"
    else:
        result["likely_db_engine"] = "unknown"
        result["restore_assessment"] = "no recognized dump header or custom archive magic"
    return result


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--short-markdown-chars", type=int, default=200)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    output = (args.output or root / "artifacts" / "data_audit").resolve()
    markdown = collect_cards(root / "data" / "cards_md", "markdown", root)
    raw = collect_cards(root / "data" / "raw_technological_cards" / "data", "raw", root)
    manifest = build_manifest(markdown, raw)
    short_markdown = [
        item | {"status": "empty" if item["non_whitespace_characters"] == 0 else "suspiciously_short"}
        for items in markdown.values() for item in items
        if item["non_whitespace_characters"] < args.short_markdown_chars
    ]
    duplicate_collisions = [
        {"source": source, "category": key[0], "normalized_title": key[1], "paths": [x["path"] for x in items]}
        for source, collection in (("markdown", markdown), ("raw", raw))
        for key, items in collection.items() if len(items) > 1
    ]
    discrepancies = {
        "raw_without_markdown": [row for row in manifest if row["has_raw"] and not row["has_markdown"]],
        "markdown_without_raw": [row for row in manifest if row["has_markdown"] and not row["has_raw"]],
        "duplicate_normalized_filenames": duplicate_collisions,
        "near_duplicate_filename_candidates": filename_candidates(markdown, "markdown") + filename_candidates(raw, "raw"),
        "empty_or_suspiciously_short_markdown": short_markdown,
    }
    dumps = [inspect_dump(path, root) for path in sorted((root / "data").rglob("*")) if path.is_file() and path.suffix.casefold() in DUMP_EXTENSIONS]
    raw_duplicate_excess = sum(max(0, len(items) - 1) for items in raw.values())
    markdown_duplicate_excess = sum(max(0, len(items) - 1) for items in markdown.values())
    summary = {
        "markdown_files": sum(len(items) for items in markdown.values()),
        "raw_files": sum(len(items) for items in raw.values()),
        "matched_card_keys": sum(row["has_markdown"] and row["has_raw"] for row in manifest),
        "raw_duplicate_excess_files": raw_duplicate_excess,
        "markdown_duplicate_excess_files": markdown_duplicate_excess,
        **{name: len(items) for name, items in discrepancies.items()},
        "database_dumps": len(dumps),
        "note": (
            "No normalized card keys are unmatched. The raw/Markdown file-count difference is explained by "
            "different numbers of files sharing one category/title key: raw duplicate excess minus Markdown "
            "duplicate excess equals the file-count difference. See duplicate_normalized_filenames for paths."
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "manifest.json", manifest)
    write_json(output / "discrepancies.json", discrepancies)
    write_json(output / "database_dumps.json", dumps)
    write_json(output / "summary.json", summary)
    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as file:
        fields = ["card_id", "category", "markdown_path", "raw_path", "raw_extension", "normalized_title", "has_markdown", "has_raw"]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in manifest)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Audit results: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
