import asyncio
import re
import urllib.parse
from pathlib import Path
from typing import Annotated, Any

import httpx
import platformdirs
from fastmcp import FastMCP
from pydantic import BaseModel, Field

MAX_FILE_SIZE_MB = 500
DEFAULT_DOWNLOAD_DIR = Path(platformdirs.user_downloads_dir()) / "mcp_downloads"

mcp = FastMCP("download")


class DownloadResult(BaseModel):
    file_path: str = Field(..., description="Full path where the file was saved")
    file_name: str = Field(..., description="Name of the downloaded file")
    file_size: int = Field(..., description="Size of the downloaded file in bytes")
    content_type: str | None = Field(
        None, description="MIME type of the downloaded file"
    )
    success: bool = Field(..., description="Whether the download was successful")
    error: str | None = Field(None, description="Error message if download failed")


def _sanitize_filename(filename: str) -> str:
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", filename)
    sanitized = sanitized.strip(". ")

    if not sanitized:
        sanitized = "downloaded_file"

    if len(sanitized) > 255:
        name, ext = Path(sanitized).stem, Path(sanitized).suffix
        sanitized = name[: 255 - len(ext)] + ext

    return sanitized


def _extract_filename_from_url(url: str) -> str:
    try:
        parsed = urllib.parse.urlparse(url)
        filename = urllib.parse.unquote(Path(parsed.path).name)

        if not filename:
            for key in ("file", "filename", "name"):
                values = urllib.parse.parse_qs(parsed.query).get(key)
                if values:
                    filename = values[0]
                    break

        if not filename:
            filename = "downloaded_file"

        if "." not in filename:
            filename = f"{filename}.bin"

        return _sanitize_filename(filename)
    except Exception:
        return "downloaded_file.bin"


def _get_unique_filepath(file_path: Path) -> Path:
    if not file_path.exists():
        return file_path

    stem = file_path.stem
    suffix = file_path.suffix
    parent = file_path.parent
    counter = 1

    while True:
        new_path = parent / f"{stem}_{counter}{suffix}"
        if not new_path.exists():
            return new_path
        counter += 1


async def _download_one(
    url: str,
    output_dir: Path,
    filename: str | None,
    timeout: int,
    max_size_bytes: int,
) -> DownloadResult:
    file_path: Path | None = None
    try:
        if not url.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")

        output_dir.mkdir(parents=True, exist_ok=True)
        resolved_name = (
            _sanitize_filename(filename)
            if filename
            else _extract_filename_from_url(url)
        )
        file_path = _get_unique_filepath(output_dir / resolved_name)

        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                content_type = response.headers.get("Content-Type")
                downloaded = 0

                with open(file_path, "wb") as f:
                    async for chunk in response.aiter_bytes(chunk_size=8192):
                        downloaded += len(chunk)
                        if downloaded > max_size_bytes:
                            raise ValueError(
                                f"File exceeded size limit "
                                f"({downloaded / (1024 * 1024):.1f} MB > {max_size_bytes / (1024 * 1024):.0f} MB)"
                            )
                        f.write(chunk)

                return DownloadResult(
                    file_path=str(file_path),
                    file_name=file_path.name,
                    file_size=file_path.stat().st_size,
                    content_type=content_type,
                    success=True,
                )

    except Exception as exc:
        if file_path and file_path.exists():
            file_path.unlink()
        return DownloadResult(
            file_path="",
            file_name=filename or "",
            file_size=0,
            success=False,
            error=str(exc),
        )


@mcp.tool
async def download(
    urls: Annotated[list[str], Field(description="URLs to download")],
    output_dir: Annotated[
        str | None, Field(description="Directory to save files")
    ] = None,
    filenames: Annotated[
        list[str] | None,
        Field(description="Custom filenames, one per URL (optional)"),
    ] = None,
    timeout: Annotated[int, Field(description="Download timeout in seconds")] = 60,
    max_size_mb: Annotated[
        int, Field(description="Maximum file size in MB")
    ] = MAX_FILE_SIZE_MB,
) -> dict[str, Any]:
    """Download one or more files from URLs to the local filesystem."""
    dest = Path(output_dir) if output_dir else DEFAULT_DOWNLOAD_DIR
    max_bytes = max_size_mb * 1024 * 1024
    names = filenames or [None] * len(urls)

    results = await asyncio.gather(
        *(
            _download_one(url, dest, name, timeout, max_bytes)
            for url, name in zip(urls, names)
        )
    )

    success_count = sum(r.success for r in results)
    return {
        "results": [r.model_dump() for r in results],
        "success_count": success_count,
        "failed_count": len(results) - success_count,
    }


def main():
    mcp.run(show_banner=False)
