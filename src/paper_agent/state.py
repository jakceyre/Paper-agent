"""LangGraph AgentState — the shared state that flows through every node.

Uses TypedDict with Annotated reducers for accumulator fields.
Without Annotated[list, add], LangGraph would overwrite list fields on each update.
"""

from operator import add
from typing import Annotated, Dict, List, Optional, Tuple, TypedDict

from paper_agent.models.paper import PaperMetadata, ParsedPaper
from paper_agent.models.claim import Claim, EvidenceItem
from paper_agent.models.comparison import ComparisonTable
from paper_agent.models.trace import TraceEvent


class AgentState(TypedDict, total=False):
    """Shared state that flows through every LangGraph node.

    Fields without a reducer use default replacement semantics.
    Fields with Annotated[list, add] use append semantics — each node's
    return dict appends to the accumulated list.
    """

    # ── Input ──────────────────────────────────────────
    query: str
    """Original user query string."""

    # ── Run identity ───────────────────────────────────
    run_id: str
    """Unique identifier for this run (12-char hex)."""

    # ── Planner output ─────────────────────────────────
    search_queries: List[str]
    """Decomposed search queries from the planner."""

    year_range: Tuple[int, int]
    """Inclusive year range, e.g. (2020, 2025)."""

    max_papers: int
    """Maximum number of papers to retrieve."""

    # ── Search results (accumulating: multi-source append) ──
    search_results: Annotated[List[PaperMetadata], add]
    """Collected paper metadata from all search sources."""

    # ── Ranker output ──────────────────────────────────
    ranked_papers: List[PaperMetadata]
    """Ranked and deduplicated subset of search_results."""

    # ── PDF processing ─────────────────────────────────
    parsed_papers: Dict[str, ParsedPaper]
    """Mapping from paper_id to parsed PDF content (sections, pages)."""

    # ── Claims (accumulating) ──────────────────────────
    claims: Annotated[List[Claim], add]
    """Extracted claims across all papers. Each claim carries paper_id + evidence."""

    # ── Comparison ─────────────────────────────────────
    comparison: Optional[ComparisonTable]
    """Structured comparison table (populated when >= 2 papers)."""

    # ── Evidence store (accumulating) ──────────────────
    evidence: Annotated[List[EvidenceItem], add]
    """Collected evidence items (appended incrementally)."""

    # ── Synthesis ──────────────────────────────────────
    review_draft: str
    """Draft review markdown from the synthesizer."""

    final_review: str
    """Polished review after the reviewer node."""

    bibtex_entries: List[str]
    """Collected BibTeX entries for cited papers."""

    # ── Control flow ───────────────────────────────────
    status: str
    """Current workflow status: 'running' | 'no_results' | 'complete' | 'error'."""

    # ── Errors (accumulating) ──────────────────────────
    errors: Annotated[List[str], add]
    """Non-fatal errors and warnings collected during the run."""

    # ── Trace (accumulating) ───────────────────────────
    trace: Annotated[List[TraceEvent], add]
    """Observability trace: timestamped events for each step."""


def initial_state(query: str, **overrides) -> AgentState:
    """Build the initial AgentState for a new run.

    Args:
        query: The user's research question.
        **overrides: Optional overrides for year_range, max_papers, etc.

    Returns:
        A complete initial AgentState dictionary.
    """
    import uuid
    from datetime import datetime, timezone

    run_id = uuid.uuid4().hex[:12]
    now = datetime.now(timezone.utc).isoformat()

    state: AgentState = {
        "query": query,
        "run_id": run_id,
        "search_queries": [],
        "year_range": overrides.get("year_range", (2020, 2025)),
        "max_papers": overrides.get("max_papers", 20),
        "search_results": [],
        "ranked_papers": [],
        "parsed_papers": {},
        "claims": [],
        "comparison": None,
        "evidence": [],
        "review_draft": "",
        "final_review": "",
        "bibtex_entries": [],
        "status": "running",
        "errors": [],
        "trace": [
            TraceEvent(
                timestamp=now,
                step="init",
                event_type="start",
                data={"query": query, "run_id": run_id},
            )
        ],
    }
    return state
