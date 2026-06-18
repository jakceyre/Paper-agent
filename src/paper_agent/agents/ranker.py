"""Ranker agent: deduplicates and ranks search results.

Reads from state.search_results, writes to state.ranked_papers.
"""

from __future__ import annotations

from datetime import datetime, timezone

from paper_agent.state import AgentState
from paper_agent.models.trace import TraceEvent


async def rank(state: AgentState) -> dict:
    """Deduplicate and rank search results.

    1. Merge results from multiple sources (arXiv, Semantic Scholar).
    2. Deduplicate by title similarity and DOI.
    3. Rank by relevance to query + citation count.
    4. Limit to max_papers.

    For MVP skeleton: simple title-based dedup, no ranking.
    """
    now = datetime.now(timezone.utc).isoformat()
    results = state.get("search_results", [])
    max_papers = state.get("max_papers", 20)

    # Dedup by title
    ranked = _deduplicate_by_title(results)[:max_papers]

    status = "no_results" if len(ranked) == 0 else state.get("status", "running")

    return {
        "ranked_papers": ranked,
        "status": status,
        "trace": [
            TraceEvent(
                timestamp=now,
                step="rank",
                event_type="end",
                data={
                    "before_dedup": len(results),
                    "after_dedup": len(ranked),
                },
            )
        ],
    }


def _deduplicate_by_title(papers: list) -> list:
    """Simple dedup by normalized title prefix."""
    seen: set[str] = set()
    unique = []
    for p in papers:
        key = p.title.lower()[:80].strip()
        if key and key not in seen:
            seen.add(key)
            unique.append(p)
    return unique
