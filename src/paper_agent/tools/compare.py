"""Paper comparison tool: build structured comparison across papers.

Uses LLM to analyze claims from multiple papers and produce a ComparisonTable
with dimensions like Method, Dataset, Metric, etc.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from paper_agent.llm.client import get_llm
from paper_agent.state import AgentState
from paper_agent.models.comparison import ComparisonTable, ComparisonRow
from paper_agent.models.trace import TraceEvent

logger = logging.getLogger(__name__)

COMPARISON_SYSTEM = """\
You are an academic paper comparison specialist. Given extracted claims from \
multiple papers, build a structured comparison table.

For each shared dimension (Method, Dataset, Metric, Backbone, Task, etc.), \
create one row per paper describing that paper's approach or result on that \
dimension.

Guidelines:
- Only compare on dimensions where all papers have relevant claims.
- If papers use different tasks or incompatible datasets, note this as a caveat.
- Be concise: each cell value should be under 100 characters.
- Cite the specific claim text as evidence.

Output as JSON:
{"dimensions": ["Method", "Dataset", "Metric"],
 "rows": [{"dimension": "Method", "paper_id": "2301.001", "value": "BERT-based retriever"},
          {"dimension": "Method", "paper_id": "2301.002", "value": "T5-based retriever"}],
 "caveats": "Different datasets used; metrics not directly comparable."}"""


async def compare_papers(state: AgentState) -> dict:
    """Build a structured comparison table across multiple papers.

    Groups claims by dimension (method, dataset, metric, etc.) and
    aligns them across papers via LLM. Flags dimensions where comparison
    is invalid (different tasks/datasets).

    Returns partial state dict with:
        - comparison: ComparisonTable or None
        - errors: any failures
        - trace: comparison event
    """
    now = datetime.now(timezone.utc).isoformat()
    claims = state.get("claims", [])
    ranked = state.get("ranked_papers", [])
    errors: list[str] = []

    paper_ids = list(dict.fromkeys(c.paper_id for c in claims))
    if len(paper_ids) < 2:
        return {
            "comparison": None,
            "errors": errors,
            "trace": [
                TraceEvent(
                    timestamp=now,
                    step="compare",
                    event_type="info",
                    data={"skipped": True, "reason": "fewer than 2 papers with claims"},
                )
            ],
        }

    fallback_used = False
    try:
        llm = get_llm()

        if not llm.available:
            logger.info("LLM not available, using stub comparison")
            comparison = _stub_comparison(paper_ids)
            fallback_used = True
        else:
            prompt = _build_comparison_prompt(claims, ranked, paper_ids)
            result = await llm.generate_with_json(
                system=COMPARISON_SYSTEM,
                prompt=prompt,
                max_tokens=2048,
            )

            if result.get("_parse_error"):
                logger.warning("Comparison JSON parse error, using stub")
                comparison = _stub_comparison(paper_ids)
                fallback_used = True
            else:
                comparison = ComparisonTable(
                    dimensions=result.get("dimensions", ["Method", "Dataset", "Metric"]),
                    paper_ids=paper_ids,
                    rows=[
                        ComparisonRow(
                            dimension=r["dimension"],
                            paper_id=r["paper_id"],
                            value=r["value"],
                        )
                        for r in result.get("rows", [])
                        if r.get("dimension") and r.get("paper_id") and r.get("value")
                    ],
                    caveats=result.get("caveats"),
                )
    except Exception as e:
        logger.error("Comparison LLM call failed: %s", e)
        errors.append(f"Comparison failed: {e}")
        comparison = _stub_comparison(paper_ids)
        fallback_used = True

    return {
        "comparison": comparison,
        "errors": errors,
        "trace": [
            TraceEvent(
                timestamp=now,
                step="compare",
                event_type="end",
                data={
                    "papers_compared": len(paper_ids),
                    "dimensions": comparison.dimensions,
                    "rows_count": len(comparison.rows),
                    "fallback_used": fallback_used,
                },
            )
        ],
    }


def _build_comparison_prompt(claims, ranked, paper_ids) -> str:
    """Build a detailed prompt for comparison from claims and paper metadata."""
    parts: list[str] = []
    parts.append("Compare the following papers based on extracted claims.")
    parts.append("")

    # Paper info
    paper_map = {p.paper_id: p for p in ranked}
    for pid in paper_ids:
        pm = paper_map.get(pid)
        title = pm.title if pm else pid
        parts.append(f"## {title} ({pid})")
        parts.append("")
        paper_claims = [c for c in claims if c.paper_id == pid]
        for c in paper_claims:
            parts.append(f"- [{c.claim_type}] {c.claim_text}")
            if c.evidence:
                parts.append(f"  Quote: {c.evidence[:150]}")
        parts.append("")

    parts.append("---")
    parts.append("Produce the comparison JSON following the system prompt format.")
    return "\n".join(parts)


def _stub_comparison(paper_ids: list[str]) -> ComparisonTable:
    """Minimal stub comparison when LLM is unavailable."""
    return ComparisonTable(
        dimensions=["Method", "Dataset", "Metric"],
        paper_ids=paper_ids,
        rows=[],
        caveats="Comparison requires LLM — set ANTHROPIC_API_KEY to populate.",
    )
