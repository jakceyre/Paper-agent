"""Paper search tools: arXiv API + Semantic Scholar API.

Both APIs are called concurrently per query. Each source operates
independently — one source failing does not block the other.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from paper_agent.state import AgentState
from paper_agent.models.paper import PaperMetadata
from paper_agent.models.trace import TraceEvent

logger = logging.getLogger(__name__)


async def search_papers(state: AgentState) -> dict:
    """Multi-source paper search: arXiv + Semantic Scholar.

    Reads from state:
        - search_queries: decomposed queries from planner
        - year_range: (from_year, to_year)
        - max_papers: cap on total results

    Appends to state (via Annotated[list, add] reducer):
        - search_results: new PaperMetadata entries
        - errors: any non-fatal API errors
        - trace: one event per source searched
    """
    now = datetime.now(timezone.utc).isoformat()
    queries = state.get("search_queries", [state["query"]])
    year_from, year_to = state["year_range"]
    max_papers = state.get("max_papers", 20)

    errors: list[str] = []
    all_results: list[PaperMetadata] = []
    source_stats: dict[str, int] = {}

    coros: list[tuple[str, asyncio.Task]] = []

    for q in queries:
        per_source = max(5, max_papers // (len(queries) * 2))
        coros.append(("arxiv", _search_arxiv(q, year_from, year_to, per_source)))
        coros.append(("s2", _search_s2(q, year_from, year_to, per_source)))

    tasks = [asyncio.ensure_future(coro) for _, coro in coros]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for (source, _), result in zip(coros, results):
        if isinstance(result, Exception):
            msg = f"{source} search failed: {result}"
            logger.warning(msg)
            errors.append(msg)
        else:
            all_results.extend(result)
            source_stats[source] = source_stats.get(source, 0) + len(result)

    # Deduplicate by paper_id
    seen: set[str] = set()
    unique: list[PaperMetadata] = []
    for p in all_results:
        if p.paper_id not in seen:
            seen.add(p.paper_id)
            unique.append(p)

    unique = unique[:max_papers]

    return {
        "search_results": unique,
        "errors": errors,
        "trace": [
            TraceEvent(
                timestamp=now,
                step="search",
                event_type="end",
                data={
                    "papers_found": len(unique),
                    "total_before_dedup": len(all_results),
                    "sources": source_stats,
                    "queries_used": len(queries),
                },
            )
        ],
    }


async def _search_arxiv(
    query: str,
    year_from: int,
    year_to: int,
    max_results: int,
) -> list[PaperMetadata]:
    """Search arXiv API synchronously via asyncio.to_thread."""
    import arxiv

    def _run():
        papers = []
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
            sort_order=arxiv.SortOrder.Descending,
        )
        for result in client.results(search):
            year = result.published.year if result.published else None
            if year is not None and (year < year_from or year > year_to):
                continue
            papers.append(
                PaperMetadata(
                    paper_id=_arxiv_id_from_entry(result.entry_id),
                    source="arxiv",
                    title=result.title.strip(),
                    authors=[a.name for a in result.authors],
                    year=year,
                    abstract=result.summary.replace("\n", " ").strip(),
                    url=result.entry_id,
                    pdf_url=result.pdf_url,
                    citation_count=None,
                    venue=result.journal_ref or None,
                )
            )
        return papers

    return await asyncio.to_thread(_run)


def _arxiv_id_from_entry(entry_id: str) -> str:
    """Extract short arXiv ID from the full entry_id URL."""
    parts = entry_id.rstrip("/").split("/")
    raw = parts[-1]
    if "v" in raw:
        v_idx = raw.rindex("v")
        if v_idx > 0 and raw[v_idx + 1:].isdigit():
            raw = raw[:v_idx]
    return raw


async def _search_s2(
    query: str,
    year_from: int,
    year_to: int,
    max_results: int,
) -> list[PaperMetadata]:
    """Search Semantic Scholar API asynchronously."""
    from semanticscholar import AsyncSemanticScholar

    papers: list[PaperMetadata] = []

    sch = AsyncSemanticScholar()
    year_str = f"{year_from}-{year_to}" if year_from != year_to else str(year_from)
    results = await sch.search_paper(
        query=query,
        year=year_str,
        limit=min(max_results, 100),
        fields=[
            "paperId", "title", "abstract", "authors", "year",
            "url", "venue", "citationCount", "openAccessPdf", "externalIds",
        ],
    )

    items = results.data if hasattr(results, "data") else [results] if results else []

    for item in items:
        if item is None:
            continue
        arxiv_id = None
        if item.externalIds and item.externalIds.get("ArXiv"):
            arxiv_id = item.externalIds["ArXiv"]
        paper_id = arxiv_id or item.paperId or ""
        if not paper_id:
            continue

        pdf_url = None
        if item.openAccessPdf and item.openAccessPdf.get("url"):
            pdf_url = item.openAccessPdf["url"]

        authors = (
            [a.get("name", "") for a in item.authors]
            if isinstance(item.authors, list)
            and all(isinstance(a, dict) for a in item.authors)
            else []
        )

        papers.append(
            PaperMetadata(
                paper_id=paper_id,
                source="semantic_scholar",
                title=item.title or "",
                authors=authors,
                year=item.year,
                abstract=item.abstract or "",
                url=item.url or f"https://www.semanticscholar.org/paper/{paper_id}",
                pdf_url=pdf_url,
                citation_count=item.citationCount,
                venue=item.venue or None,
            )
        )

    return papers


async def expand_citations(
    paper_id: str,
    depth: int = 1,
    *,
    state: AgentState | None = None,
) -> dict:
    """Expand the citation graph for a given paper via Semantic Scholar."""
    return {
        "search_results": [],
        "errors": [],
    }
