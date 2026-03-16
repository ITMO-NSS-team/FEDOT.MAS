from __future__ import annotations

import zipfile
from pathlib import Path

import pytest
from fastmcp import Client

from mcp_document.server import _truncate, mcp


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
