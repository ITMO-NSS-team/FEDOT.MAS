"""Opt-in HTTP transport for OpenRouter, leaving other services untouched."""

from __future__ import annotations

import os
from urllib.parse import urlsplit

import httpx


def is_openrouter_url(base_url: str | None) -> bool:
    """Only the HTTPS OpenRouter API is eligible for the dedicated proxy."""
    if not base_url:
        return False
    try:
        url = urlsplit(base_url)
        return url.scheme == "https" and url.hostname == "openrouter.ai"
    except ValueError:
        return False


def openrouter_http_client(base_url: str | None) -> httpx.AsyncClient | None:
    """Return a proxied client only for OpenRouter when explicitly configured."""
    proxy_url = os.getenv("FEDOTMAS_OPENROUTER_PROXY_URL", "").strip()
    if not proxy_url or not is_openrouter_url(base_url):
        return None
    return httpx.AsyncClient(proxy=proxy_url, follow_redirects=True)
