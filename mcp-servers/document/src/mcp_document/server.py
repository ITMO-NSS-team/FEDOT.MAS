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
    model = os.getenv("DOCUMENT_VISION_MODEL", "google/gemini-2.5-flash")
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


class DocumentResult(BaseModel):
    content: str = Field(default="", description="Extracted text in markdown")
    error: str | None = Field(default=None, description="Error message if failed")


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
    describe_images: Annotated[
        bool, Field(description="Use LLM vision to describe embedded images")
    ] = False,
) -> DocumentResult:
    """Read a document and extract text content as markdown.

    Supports: PDF, DOCX, PPTX, XLSX, CSV, JSON, XML, HTML,
    plain text, and source code files.

    Set describe_images=True to use LLM vision for embedded images
    (requires OPENAI_API_KEY). Without it, images are skipped or shown as alt text.
    """
    try:
        path = os.path.expanduser(file_path)
        await ctx.info(f"Reading: {os.path.basename(path)}")
        md = _md_with_vision() if describe_images else _md
        content = md.convert(path).text_content
        return DocumentResult(content=_truncate(content, max_lines))
    except Exception as e:
        await ctx.error(f"Failed to read document: {e}")
        return DocumentResult(error=str(e))


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
    except Exception as e:
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
    except Exception as e:
        await ctx.error(f"Failed to extract ZIP: {e}")
        return ZipExtractResult(error=str(e))


def main():
    mcp.run(show_banner=False)
