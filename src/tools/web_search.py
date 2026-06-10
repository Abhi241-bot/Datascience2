"""Web search tool — returns results with URLs (provenance for citations).

Uses Tavily when `TAVILY_API_KEY` is set (better for agents); otherwise falls back
to free DuckDuckGo. Always returns a uniform structure so the agent can cite URLs.

Standalone use:
    from src.tools.web_search import web_search
    print(web_search("Nimbus Cloud 2024 cloud revenue guidance"))
"""
from __future__ import annotations

from typing import Any

from src import config


def _search_tavily(query: str, max_results: int) -> list[dict]:
    from tavily import TavilyClient

    client = TavilyClient(api_key=config.TAVILY_API_KEY)
    resp = client.search(query=query, max_results=max_results, search_depth="basic")
    return [
        {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")}
        for r in resp.get("results", [])
    ]


def _search_duckduckgo(query: str, max_results: int) -> list[dict]:
    from duckduckgo_search import DDGS

    out = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            out.append(
                {
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                }
            )
    return out


def web_search(query: str, max_results: int | None = None) -> dict[str, Any]:
    """Run a web search; returns {query, results:[{title,url,snippet}], provider}."""
    max_results = max_results or config.WEB_SEARCH_MAX_RESULTS
    provider = "tavily" if config.TAVILY_API_KEY else "duckduckgo"
    try:
        if provider == "tavily":
            results = _search_tavily(query, max_results)
        else:
            results = _search_duckduckgo(query, max_results)
        error = None
    except Exception as e:
        results, error = [], f"web search failed: {e}"

    return {
        "query": query,
        "results": results,
        "provider": provider,
        "error": error,
        "source": "web_search",
    }


if __name__ == "__main__":
    import json

    print(json.dumps(web_search("Nimbus Cloud enterprise cloud revenue growth 2024"), indent=2))
