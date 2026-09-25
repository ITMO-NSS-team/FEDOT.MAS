from __future__ import annotations

import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastmcp import Client
from mcp_document.server import (
    _page_document,
    _truncate,
    find_document,
    mcp,
    read_document,
)


class TestTruncate:
    def test_none_max_lines_returns_as_is(self):
        text = "a\nb\nc"
        assert _truncate(text, None) == text

    def test_within_limit_returns_as_is(self):
        text = "a\nb\nc"
        assert _truncate(text, 5) == text

    def test_exact_limit_returns_as_is(self):
        text = "a\nb\nc"
        assert _truncate(text, 3) == text

    def test_over_limit_truncates_with_indicator(self):
        text = "line1\nline2\nline3\nline4\nline5"
        result = _truncate(text, 2)
        assert result.startswith("line1\nline2\n")
        assert "truncated" in result
        assert "2/5" in result


def test_document_paging_exposes_all_large_xml_characters():
    text = "<root>\n" + "<item>early value</item>\n" * 500 + "<final>late sentinel</final>\n</root>"
    page, total_lines, _, next_line, next_char, truncated = _page_document(
        text,
        start_line=0,
        max_lines=1000,
        start_char=None,
        max_chars=700,
    )
    collected = page
    while truncated:
        page, _, _start_line, next_line, next_char, truncated = _page_document(
            text,
            start_line=next_line,
            max_lines=1000,
            start_char=next_char,
            max_chars=700,
        )
        collected += page

    assert total_lines > 500
    assert "late sentinel" in collected


@pytest.fixture
def _txt_file(tmp_path: Path) -> Path:
    f = tmp_path / "hello.txt"
    f.write_text("Hello, world!\nSecond line.")
    return f


@pytest.fixture
def _zip_file(tmp_path: Path) -> Path:
    zp = tmp_path / "test.zip"
    with zipfile.ZipFile(zp, "w") as zf:
        zf.writestr("a.txt", "content a")
        zf.writestr("subdir/b.txt", "content b")
    return zp


def _text(result) -> str:
    return result.content[0].text


class TestReadDocument:
    @pytest.mark.anyio
    async def test_reads_txt_file(self, _txt_file: Path):
        async with Client(mcp) as c:
            result = await c.call_tool("read_document", {"file_path": str(_txt_file)})
            assert "Hello, world!" in _text(result)

    @pytest.mark.anyio
    async def test_missing_file_returns_error(self, tmp_path: Path):
        async with Client(mcp) as c:
            result = await c.call_tool(
                "read_document", {"file_path": str(tmp_path / "nope.txt")}
            )
            text = _text(result)
            assert "error" in text.lower() or "no such file" in text.lower()

    @pytest.mark.anyio
    async def test_large_xml_can_be_read_after_the_first_chunk(self, tmp_path: Path):
        path = tmp_path / "large.xml"
        path.write_text(
            "<root>\n"
            + "<item>ordinary</item>\n" * 700
            + "<final>late sentinel</final>\n</root>",
            encoding="utf-8",
        )
        ctx = MagicMock()
        ctx.info = AsyncMock()
        ctx.error = AsyncMock()
        cursor = None
        found = False
        for _ in range(20):
            page = await read_document(
                str(path),
                ctx,
                start_char=cursor,
                max_chars=1000,
            )
            found = found or "late sentinel" in page.content
            cursor = page.next_start_char
            if cursor is None:
                break

        assert found is True
        assert page.total_lines > 700

    @pytest.mark.anyio
    async def test_find_document_returns_targeted_context(self, tmp_path: Path):
        path = tmp_path / "large.xml"
        path.write_text("<root>\n<item>late-value-42</item>\n</root>", encoding="utf-8")
        ctx = MagicMock()
        ctx.info = AsyncMock()
        ctx.error = AsyncMock()

        result = await find_document(str(path), "late-value-42", ctx)

        assert result.total_matches == 1
        assert result.matches[0].line == 2
        assert "late-value-42" in result.matches[0].text

    @pytest.mark.anyio
    async def test_find_document_paginates_later_matches(self, tmp_path: Path):
        path = tmp_path / "many.xml"
        path.write_text(
            "<root>\n"
            + "".join(f"<item>target-{index}</item>\n" for index in range(12))
            + "</root>",
            encoding="utf-8",
        )
        ctx = MagicMock()
        ctx.info = AsyncMock()
        ctx.error = AsyncMock()

        first = await find_document(str(path), "target-", ctx)
        second = await find_document(
            str(path), "target-", ctx, start_line=first.next_start_line
        )

        assert len(first.matches) == 10
        assert first.next_start_line == first.matches[-1].line + 1
        assert [match.line for match in second.matches] == [12, 13]
        assert second.next_start_line is None
        assert second.total_matches == 12

    @pytest.mark.anyio
    async def test_reads_legacy_xls_with_xlrd(self, tmp_path: Path, monkeypatch):
        class FakeSheet:
            name = "Records"
            nrows = 1
            ncols = 2

            @staticmethod
            def cell_value(row: int, column: int):
                return [["name", "orcid"]][row][column]

        class FakeWorkbook:
            @staticmethod
            def sheets():
                return [FakeSheet()]

            @staticmethod
            def release_resources():
                return None

        monkeypatch.setitem(
            sys.modules,
            "xlrd",
            SimpleNamespace(open_workbook=lambda *_args, **_kwargs: FakeWorkbook()),
        )
        path = tmp_path / "legacy.xls"
        path.write_bytes(b"test fixture")
        ctx = MagicMock()
        ctx.info = AsyncMock()
        ctx.error = AsyncMock()

        result = await read_document(str(path), ctx)

        assert "## Sheet: Records" in result.content
        assert "1\tname\torcid" in result.content


class TestZip:
    @pytest.mark.anyio
    async def test_list_zip_contents(self, _zip_file: Path):
        async with Client(mcp) as c:
            result = await c.call_tool(
                "list_zip_contents", {"file_path": str(_zip_file)}
            )
            text = _text(result)
            assert "a.txt" in text
            assert "b.txt" in text

    @pytest.mark.anyio
    async def test_extract_zip(self, _zip_file: Path, tmp_path: Path):
        dest = tmp_path / "out"
        async with Client(mcp) as c:
            result = await c.call_tool(
                "extract_zip",
                {"file_path": str(_zip_file), "output_dir": str(dest)},
            )
            assert "a.txt" in _text(result)
        assert (dest / "a.txt").read_text() == "content a"
        assert (dest / "subdir" / "b.txt").read_text() == "content b"
