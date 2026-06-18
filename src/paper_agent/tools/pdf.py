"""PDF tools: download and parse academic papers."""

from __future__ import annotations

from datetime import datetime, timezone

from paper_agent.state import AgentState
from paper_agent.models.trace import TraceEvent


async def download_pdf(state: AgentState) -> dict:
    """Download PDFs for all papers in state.ranked_papers."""
    now = datetime.now(timezone.utc).isoformat()
    parsed_papers: dict = {}
    errors: list[str] = []

    return {
        "parsed_papers": parsed_papers,
        "errors": errors,
        "trace": [
            TraceEvent(
                timestamp=now,
                step="fetch_pdfs",
                event_type="end",
                data={"downloaded": len(parsed_papers), "failed": len(errors)},
            )
        ],
    }


async def parse_pdf(state: AgentState) -> dict:
    """Parse downloaded PDFs into structured sections with page numbers.

    Reads state.parsed_papers (populated by download_pdf with local_pdf_path),
    parses each PDF with PyMuPDF, and returns the updated dict.
    """
    now = datetime.now(timezone.utc).isoformat()
    parsed_papers: dict = dict(state.get("parsed_papers", {}))
    errors: list[str] = []

    return {
        "parsed_papers": parsed_papers,
        "errors": errors,
        "trace": [
            TraceEvent(
                timestamp=now,
                step="parse_pdfs",
                event_type="end",
                data={"parsed": len(parsed_papers), "failed": len(errors)},
            )
        ],
    }
