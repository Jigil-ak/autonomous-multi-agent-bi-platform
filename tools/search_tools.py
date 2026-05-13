"""
Search Tools
=============
DuckDuckGo-based search tools for the Research Agent.

Design decisions:
- sleep(2) between requests to avoid temporary DuckDuckGo rate-limiting
- All failures are caught and logged — never crashes the calling agent
- Returns structured list of result dicts for deterministic downstream use
- Mock fallback data provided for demo resilience when search is unavailable
"""

from __future__ import annotations

import time
from typing import Any

from duckduckgo_search import DDGS

from config import SEARCH_DELAY_SECONDS
from utils.logger import get_logger

logger = get_logger("search_tools")

# ---------------------------------------------------------------------------
# Mock search results (used when DuckDuckGo is unavailable / rate-limited)
# ---------------------------------------------------------------------------
_MOCK_RESULTS: dict[str, list[dict[str, str]]] = {
    "default": [
        {
            "title": "Market Overview: AI-Powered Business Intelligence",
            "body": "The global BI market is expected to grow at 10.1% CAGR through 2028. AI-first platforms are capturing significant market share.",
            "href": "https://example.com/bi-market-report",
        },
        {
            "title": "Competitor Landscape in Modern BI Tools",
            "body": "Key players include Tableau, Power BI, Looker, and emerging AI-native startups. Pricing ranges from $15 to $500+ per seat per month.",
            "href": "https://example.com/bi-competitors",
        },
        {
            "title": "SMB Analytics Trends 2025",
            "body": "Small businesses are increasingly adopting self-service analytics. Key drivers: cost reduction, faster decisions, no-code interfaces.",
            "href": "https://example.com/smb-analytics-2025",
        },
    ]
}


# ---------------------------------------------------------------------------
# Core search function
# ---------------------------------------------------------------------------
def search_web(
    query: str,
    max_results: int = 5,
    use_mock_on_failure: bool = True,
) -> list[dict[str, str]]:
    """
    Search DuckDuckGo and return structured results.

    Parameters
    ----------
    query : str
        The search query string.
    max_results : int
        Maximum number of results to return.
    use_mock_on_failure : bool
        Return mock data if the search fails.

    Returns
    -------
    list[dict]
        List of {title, body, href} dicts.
    """
    logger.info(f"Searching: '{query}' (max_results={max_results})")

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))

        if not results:
            logger.warning(f"No results returned for query: '{query}'")
            return _MOCK_RESULTS["default"] if use_mock_on_failure else []

        logger.info(f"Search returned {len(results)} results for: '{query}'")
        return [
            {
                "title": r.get("title", ""),
                "body": r.get("body", ""),
                "href": r.get("href", ""),
            }
            for r in results
        ]

    except Exception as exc:
        logger.warning(f"Search failed for '{query}': {exc}. Using mock data.")
        return _MOCK_RESULTS["default"] if use_mock_on_failure else []


# ---------------------------------------------------------------------------
# Multi-query research pipeline
# ---------------------------------------------------------------------------
def run_research_queries(
    queries: list[str],
    max_results_per_query: int = 4,
    delay_seconds: float = SEARCH_DELAY_SECONDS,
) -> list[dict[str, Any]]:
    """
    Execute multiple search queries with rate-limit delay between each.

    Parameters
    ----------
    queries : list[str]
        List of search queries to execute.
    max_results_per_query : int
        Max results per query.
    delay_seconds : float
        Sleep time between queries to avoid blocking.

    Returns
    -------
    list[dict]
        Flat list of all search results with source query attached.
    """
    all_results: list[dict[str, Any]] = []

    for i, query in enumerate(queries):
        results = search_web(query, max_results=max_results_per_query)
        for r in results:
            r["source_query"] = query
            all_results.append(r)

        # Rate-limit delay between queries (skip delay after last query)
        if i < len(queries) - 1:
            logger.info(f"Waiting {delay_seconds}s before next search query...")
            time.sleep(delay_seconds)

    logger.info(
        f"Research pipeline complete: {len(queries)} queries, "
        f"{len(all_results)} total results."
    )
    return all_results


# ---------------------------------------------------------------------------
# Convenience helpers for common research tasks
# ---------------------------------------------------------------------------
def search_competitors(company_name: str, industry: str) -> list[dict[str, str]]:
    """Search for competitors in a specific industry."""
    query = f"{company_name} competitors alternatives {industry} 2024 2025"
    return search_web(query, max_results=5)


def search_market_trends(industry: str, product_type: str) -> list[dict[str, str]]:
    """Search for market trends and industry signals."""
    query = f"{industry} {product_type} market trends growth 2024 2025"
    return search_web(query, max_results=4)


def search_pricing(product_type: str, industry: str) -> list[dict[str, str]]:
    """Search for pricing benchmarks in a market."""
    query = f"{product_type} {industry} pricing comparison cost per user SaaS"
    return search_web(query, max_results=4)


def search_audience_insights(
    target_audience: str, product_type: str
) -> list[dict[str, str]]:
    """Search for audience behaviour and preferences."""
    query = f"{target_audience} {product_type} pain points needs preferences"
    return search_web(query, max_results=4)


# ---------------------------------------------------------------------------
# Format results as readable text block
# ---------------------------------------------------------------------------
def format_results_as_text(results: list[dict[str, Any]]) -> str:
    """
    Convert search results list into a readable text block for LLM prompts.
    """
    if not results:
        return "No search results available."

    lines: list[str] = []
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] {r.get('title', 'No title')}")
        lines.append(f"    {r.get('body', 'No content')}")
        if r.get("href"):
            lines.append(f"    Source: {r['href']}")
        lines.append("")

    return "\n".join(lines)
