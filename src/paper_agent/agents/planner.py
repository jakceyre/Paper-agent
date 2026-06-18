"""Planner agent: decomposes user query into search parameters.

This is the first LangGraph node after START.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone

from paper_agent.llm.client import get_llm
from paper_agent.state import AgentState
from paper_agent.models.trace import TraceEvent

logger = logging.getLogger(__name__)

PLANNER_SYSTEM = """\
You are an academic research search strategist. Given a research topic or question, \
your task is to decompose it into 2-3 precise search queries suitable for \
arXiv and Semantic Scholar APIs.

Guidelines:
1. Use technical, academic language — keywords from the field.
2. Include synonyms and alternative phrasings of key concepts.
3. Each query should be short (under 80 characters) and focused.
4. Avoid overly broad single-word queries.

Output format:
Return exactly one query per line, prefixed with "Q:".
Do not include explanations, numbering, or markdown formatting.

Example:
Q: retrieval augmented generation multi-hop QA
Q: RAG reasoning chain-of-thought retrieval"""


async def plan(state: AgentState) -> dict:
    """Decompose the user query into structured search parameters.

    Uses LLM to identify key concepts, generate synonyms, and produce
    2-3 targeted search queries for the paper search step.

    On LLM failure, falls back to using the raw query as-is.
    """
    now = datetime.now(timezone.utc).isoformat()
    query = state["query"]
    errors: list[str] = []
    search_queries: list[str] = []

    try:
        llm = get_llm()

        if not llm.available:
            logger.info("LLM not available, using raw query for search")
            search_queries = [query]
        else:
            response = await llm.generate(
                system=PLANNER_SYSTEM,
                messages=[{"role": "user", "content": query}],
                max_tokens=300,
                temperature=0.3,
            )
            search_queries = _parse_queries(response)

            if not search_queries:
                logger.warning("Planner returned no queries, falling back to raw query")
                search_queries = [query]
    except Exception as e:
        logger.error("Planner LLM call failed: %s", e)
        errors.append(f"Planner failed: {e}")
        search_queries = [query]

    return {
        "search_queries": search_queries,
        "errors": errors,
        "trace": [
            TraceEvent(
                timestamp=now,
                step="plan",
                event_type="end",
                data={
                    "search_queries": search_queries,
                    "year_range": list(state["year_range"]),
                    "fallback_used": len(errors) > 0,
                },
            )
        ],
    }


def _parse_queries(text: str) -> list[str]:
    """Parse LLM response into a list of search queries.

    Expected format: "Q: query text" — one per line.
    Also handles numbered/bulleted formats as fallback.
    """
    queries: list[str] = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        # Preferred: "Q: query"
        if line.upper().startswith("Q:"):
            q = line[2:].strip()
            if q and len(q) > 3:
                queries.append(q)
            continue
        # Fallback: "1. query" or "- query" or "query"
        cleaned = re.sub(r"^[\d]+[\.\)]\s*", "", line)
        cleaned = re.sub(r"^[-*•]\s*", "", cleaned)
        cleaned = cleaned.strip()
        if cleaned and len(cleaned) > 3:
            queries.append(cleaned)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for q in queries:
        norm = q.lower()
        if norm not in seen:
            seen.add(norm)
            unique.append(q)

    return unique[:5]
