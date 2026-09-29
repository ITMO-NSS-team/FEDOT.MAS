import os
from typing import Annotated, Any

import httpx
from dotenv import load_dotenv
from fastmcp import Context, FastMCP
from pydantic import BaseModel, Field

load_dotenv()


class SearchResult(BaseModel):
    url: str
    title: str
    content: str


class SearchResponse(BaseModel):
    query: str
    number_of_results: int
    results: list[SearchResult]


DESCRIPTION = """
MCP server for web search via self-hosted SearXNG. Supports filtering by category
(general, news, science, etc.) and language.
"""

ENGINES = "bing,duckduckgo,brave,mullvadleta,yahoo,presearch"
MAX_RETURNED_RESULTS = 6
MAX_TITLE_CHARS = 160
MAX_SNIPPET_CHARS = 320

searxng_server = FastMCP("websearch-searxng", instructions=DESCRIPTION)


@searxng_server.tool
async def search(
    query: Annotated[str, Field(description="Search query text")],
    ctx: Context,
    max_results: Annotated[
        int, Field(description="Maximum number of results to return", ge=1, le=100)
    ] = 10,
    categories: Annotated[
        str,
        Field(
            description="Search category: general, news, images, videos, music, files, "
            "science, social media, etc. Default: general"
        ),
    ] = "general",
    language: Annotated[
        str,
        Field(description="Language code (e.g., 'en', 'es', 'fr'). Default: auto"),
    ] = "auto",
    safesearch: Annotated[
        int,
        Field(
            description="SafeSearch level: 0 (off), 1 (moderate), 2 (strict)",
            ge=0,
            le=2,
        ),
    ] = 1,
) -> SearchResponse:
    """
    Search the web.

    Args:
        query: Search query text
        max_results: Maximum number of results (1-100)
        categories: general, news, images, videos, music, files, science, social media
        language: Language preference (e.g., "en", "es", "fr")
        safesearch: Content filtering - 0 (off), 1 (moderate), 2 (strict)

    Returns:
        SearchResponse containing:
        - query: The search query
        - number_of_results: Count of returned results
        - results: Compact SearchResult objects (url, title, content), at most 6

    Examples:
        searxng_search("breaking news AI", categories="news")
        searxng_search("python tutorials", max_results=20)
    """
    try:
        # Get SearXNG instance URL from environment, default to localhost
        instance_url = os.getenv("SEARXNG_URL", "http://localhost:18888")

        await ctx.info(f"Searching via SearXNG: {query[:50]}...")

        # Build query parameters
        params: dict[str, Any] = {
            "q": query,
            "format": "json",
            "engines": ENGINES,
            "categories": categories,
            "safesearch": safesearch,
        }

        if language != "auto":
            params["language"] = language

        # Make async request to SearXNG
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{instance_url}/search", params=params)
            response.raise_for_status()
            data = response.json()

        # Extract and limit results
        if not isinstance(data, dict) or not isinstance(data.get("results"), list):
            raise TypeError("SearXNG returned an invalid results payload")
        unresponsive = data.get("unresponsive_engines") or []
        if not data["results"] and len(unresponsive) >= len(ENGINES.split(",")):
            raise RuntimeError(
                "SearXNG search failed: all configured engines are unavailable"
            )
        raw_results = data["results"][: min(max_results, MAX_RETURNED_RESULTS)]

        # Validate and construct SearchResult objects, skip malformed entries
        results: list[SearchResult] = []
        for raw in raw_results:
            try:
                if not isinstance(raw, dict) or not raw.get("url"):
                    continue
                results.append(
                    SearchResult(
                        url=str(raw["url"]),
                        title=str(raw.get("title") or "")[:MAX_TITLE_CHARS],
                        content=str(raw.get("content") or "")[:MAX_SNIPPET_CHARS],
                    )
                )
            except (TypeError, ValueError):
                continue

        await ctx.info(f"Found {len(results)} results from SearXNG")

        return SearchResponse(
            query=query,
            number_of_results=len(results),
            results=results,
        )

    except httpx.HTTPStatusError as e:
        error_msg = (
            f"SearXNG HTTP error {e.response.status_code}: {e.response.text[:200]}"
        )
        await ctx.error(error_msg)
        raise RuntimeError(error_msg) from e
    except httpx.RequestError as e:
        error_msg = f"SearXNG connection error: {e}. Check if SearXNG is running at {instance_url}"
        await ctx.error(error_msg)
        raise RuntimeError(error_msg) from e
    except (TypeError, ValueError) as e:
        error_msg = f"SearXNG search failed: {e}"
        await ctx.error(error_msg)
        raise RuntimeError(error_msg) from e


def main():
    searxng_server.run(show_banner=False)
