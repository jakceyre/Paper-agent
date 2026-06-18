"""Paper comparison tool: build structured comparison across papers."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from paper_agent.state import AgentState
from paper_agent.models.comparison import ComparisonTable
from paper_agent.models.trace import TraceEvent

logger = logging.getLogger(__name__)


async def compare_papers(state: AgentState) -> dict:
    """Build a structured comparison table across multiple papers."""
    now = datetime.now(timezone.utc).isoformat()
    claims = state.get("claims", [])
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

    try:
        comparison = ComparisonTable(
            dimensions=["Method", "Dataset", "Metric"],
            paper_ids=paper_ids,
            rows=[],
            caveats="Comparison is preliminary; verify task and dataset compatibility.",
        )
    except Exception as e:
        logger.error("Comparison LLM call failed: %s", e)
        errors.append(f"Comparison failed: {e}")
        comparison = ComparisonTable(
            dimensions=["Method", "Dataset", "Metric"],
            paper_ids=paper_ids,
            rows=[],
            caveats=f"Comparison unavailable due to error: {e}",
        )

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
                    "fallback_used": len(errors) > 0,
                },
            )
        ],
    }
