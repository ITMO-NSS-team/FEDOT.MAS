from __future__ import annotations

import os
import zipfile
from functools import cache
from pathlib import Path
from typing import Annotated

import platformdirs
from fastmcp import Context, FastMCP
from markitdown import MarkItDown
from openai import OpenAI
from pydantic import BaseModel, Field

mcp = FastMCP("document")
_md = MarkItDown()

_CACHE_BASE = Path(platformdirs.user_cache_dir("fedotmas"))


@cache
def _md_with_vision() -> MarkItDown:
    """Lazy-init MarkItDown with LLM vision for image descriptions."""
    client = OpenAI(
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    model = os.getenv("DOCUMENT_VISION_MODEL", "google/gemini-3-flash-preview")
    return MarkItDown(llm_client=client, llm_model=model)


def _truncate(text: str, max_lines: int | None) -> str:
    if max_lines is None:
        return text
    lines = text.split("\n")
    if len(lines) <= max_lines:
        return text
    return (
        "\n".join(lines[:max_lines])
        + f"\n\n... (truncated, {max_lines}/{len(lines)} lines)"
    )


def _page_document(
    text: str,
    *,
    start_line: int,
    max_lines: int | None,
    start_char: int | None,
    max_chars: int,
) -> tuple[str, int, int, int, int, bool]:
    total_lines = text.count("\n") + (1 if text else 0)
    if start_char is not None:
        offset = min(max(0, start_char), len(text))
        page = text[offset : offset + max_chars]
    else:
        lines = text.splitlines(keepends=True)
        start_line = min(max(0, start_line), len(lines))
        selected = lines[start_line:] if max_lines is None else lines[start_line : start_line + max_lines]
        offset = sum(len(line) for line in lines[:start_line])
        page = "".join(selected)[:max_chars]
    next_char = offset + len(page)
    next_line = text[:next_char].count("\n")
    return page, total_lines, (text[:offset].count("\n")), next_line, next_char, next_char < len(text)


def _convert_xls(path: str) -> str:
    """Convert legacy XLS sheets to a compact, pageable text representation."""
    import xlrd

    workbook = xlrd.open_workbook(path, on_demand=True)
    lines = []
    for sheet in workbook.sheets():
        lines.append(f"## Sheet: {sheet.name}")
        for row_index in range(sheet.nrows):
            values = [str(sheet.cell_value(row_index, column)) for column in range(sheet.ncols)]
            lines.append(f"{row_index + 1}\t" + "\t".join(values))
    workbook.release_resources()
    return "\n".join(lines)


def _extract_document(path: str, describe_images: bool = False) -> str:
    if Path(path).suffix.lower() == ".xls":
        return _convert_xls(path)
    md = _md_with_vision() if describe_images else _md
    return md.convert(path).text_content


class DocumentResult(BaseModel):
    content: str = Field(default="", description="Extracted text in markdown")
    error: str | None = Field(default=None, description="Error message if failed")
    total_lines: int = 0
    start_line: int = 0
    next_start_line: int | None = None
    total_chars: int = 0
    next_start_char: int | None = None
    truncated: bool = False


class DocumentMatch(BaseModel):
    line: int
    text: str


class DocumentFindResult(BaseModel):
    matches: list[DocumentMatch] = Field(default_factory=list)
    total_matches: int = 0
    next_start_line: int | None = None
    error: str | None = None


class ZipEntry(BaseModel):
    name: str
    size: int
    is_dir: bool


class ZipContentsResult(BaseModel):
    entries: list[ZipEntry] = Field(default_factory=list)
    total_files: int = 0
    total_size: int = 0
    error: str | None = None


class ZipExtractResult(BaseModel):
    output_dir: str = ""
    files: list[str] = Field(default_factory=list)
    total_extracted: int = 0
    error: str | None = None


@mcp.tool
async def read_document(
    file_path: Annotated[str, Field(description="Path to the document")],
    ctx: Context,
    max_lines: Annotated[int | None, Field(description="Max lines to return")] = 1000,
    start_line: Annotated[int, Field(description="First line to return", ge=0)] = 0,
    start_char: Annotated[int | None, Field(description="Character cursor from a prior page", ge=0)] = None,
    max_chars: Annotated[int, Field(description="Maximum page size in characters", ge=500, le=4500)] = 3500,
    describe_images: Annotated[
        bool, Field(description="Use LLM vision to describe embedded images")
    ] = False,
) -> DocumentResult:
    """Read a document and extract text content as markdown.

    Supports: PDF, DOCX, PPTX, XLS, XLSX, CSV, JSON, XML, HTML,
    plain text, and source code files.

    Use start_line/max_lines or the returned next_start_char cursor to read later
    pages. Each response is bounded so result transport truncation cannot hide the
    next page. Set describe_images=True to use LLM vision for embedded images
    (requires OPENAI_API_KEY). Without it, images are skipped or shown as alt text.
    """
    try:
        path = os.path.expanduser(file_path)
        await ctx.info(f"Reading: {os.path.basename(path)}")
        content = _extract_document(path, describe_images)
        page, total_lines, actual_start_line, next_line, next_char, truncated = _page_document(
            content,
            start_line=start_line,
            max_lines=max_lines,
            start_char=start_char,
            max_chars=max_chars,
        )
        return DocumentResult(
            content=page,
            total_lines=total_lines,
            start_line=actual_start_line,
            next_start_line=next_line if truncated else None,
            total_chars=len(content),
            next_start_char=next_char if truncated else None,
            truncated=truncated,
        )
    except Exception as e:  # noqa: BLE001 - surface document conversion errors via MCP
        await ctx.error(f"Failed to read document: {e}")
        return DocumentResult(error=str(e))


@mcp.tool
async def find_document(
    file_path: Annotated[str, Field(description="Path to the document")],
    query: Annotated[str, Field(description="Text to find")],
    ctx: Context,
    start_line: Annotated[int, Field(description="First 1-based line to search", ge=1)] = 1,
    max_matches: Annotated[int, Field(description="Maximum matching lines to return", ge=1, le=10)] = 10,
    context_chars: Annotated[int, Field(description="Characters around each match", ge=0, le=200)] = 160,
) -> DocumentFindResult:
    """Find text in a document and return matching lines with a continuation cursor."""
    try:
        path = os.path.expanduser(file_path)
        await ctx.info(f"Searching: {os.path.basename(path)}")
        content = _extract_document(path)
        needle = query.casefold()
        if not needle:
            return DocumentFindResult(error="query must not be empty")
        lines = content.splitlines()
        matches: list[DocumentMatch] = []
        total = 0
        for line_number, line in enumerate(lines, start=1):
            position = line.casefold().find(needle)
            if position < 0:
                continue
            total += 1
            if line_number < start_line or len(matches) >= max_matches:
                continue
            left = max(0, position - context_chars)
            right = min(len(line), position + len(query) + context_chars)
            matches.append(DocumentMatch(line=line_number, text=line[left:right]))
        has_more = any(
            line_number > matches[-1].line and needle in line.casefold()
            for line_number, line in enumerate(lines, start=1)
        ) if matches else any(
            line_number >= start_line and needle in line.casefold()
            for line_number, line in enumerate(lines, start=1)
        )
        return DocumentFindResult(
            matches=matches,
            total_matches=total,
            next_start_line=matches[-1].line + 1 if has_more and matches else None,
        )
    except Exception as e:  # noqa: BLE001 - surface document search errors via MCP
        await ctx.error(f"Failed to search document: {e}")
        return DocumentFindResult(error=str(e))


@mcp.tool
async def list_zip_contents(
    file_path: Annotated[str, Field(description="Path to ZIP file")],
    ctx: Context,
) -> ZipContentsResult:
    """List contents of a ZIP archive without extracting."""
    try:
        path = os.path.expanduser(file_path)
        await ctx.info(f"Listing: {os.path.basename(path)}")

        entries, total_size = [], 0
        with zipfile.ZipFile(path, "r") as zf:
            for info in zf.infolist():
                is_dir = info.filename.endswith("/")
                entries.append(
                    ZipEntry(
                        name=info.filename,
                        size=info.file_size,
                        is_dir=is_dir,
                    )
                )
                if not is_dir:
                    total_size += info.file_size

        files = [e for e in entries if not e.is_dir]
        return ZipContentsResult(
            entries=entries,
            total_files=len(files),
            total_size=total_size,
        )
    except Exception as e:  # noqa: BLE001 - preserve structured ZIP tool errors
        await ctx.error(f"Failed to list ZIP: {e}")
        return ZipContentsResult(error=str(e))


@mcp.tool
async def extract_zip(
    file_path: Annotated[str, Field(description="Path to ZIP file")],
    ctx: Context,
    output_dir: Annotated[str | None, Field(description="Output directory")] = None,
) -> ZipExtractResult:
    """Extract a ZIP archive and return list of extracted files."""
    try:
        path = os.path.expanduser(file_path)
        if output_dir:
            dest = Path(os.path.expanduser(output_dir))
        else:
            dest = _CACHE_BASE / "zip" / Path(path).stem
        dest.mkdir(parents=True, exist_ok=True)

        await ctx.info(f"Extracting: {os.path.basename(path)} → {dest}")

        with zipfile.ZipFile(path, "r") as zf:
            zf.extractall(dest)
            names = [n for n in zf.namelist() if not n.endswith("/")]

        return ZipExtractResult(
            output_dir=str(dest),
            files=names,
            total_extracted=len(names),
        )
    except Exception as e:  # noqa: BLE001 - preserve structured ZIP tool errors
        await ctx.error(f"Failed to extract ZIP: {e}")
        return ZipExtractResult(error=str(e))


def main():
    mcp.run(show_banner=False)
