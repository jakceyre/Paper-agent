"""Synthesizer agent: drafts the review from collected evidence.

Uses LLM to generate a structured survey from paper metadata and claims.
Falls back to deterministic Markdown rendering on LLM failure.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from paper_agent.llm.client import get_llm
from paper_agent.state import AgentState
from paper_agent.models.trace import TraceEvent
from paper_agent.tools.write_review import render_review_markdown

logger = logging.getLogger(__name__)

SYNTHESIS_SYSTEM = """\
You are an academic survey writer. Given a collection of papers and extracted \
claims, write a structured literature survey in Markdown.

Follow this structure:
1. **Key Findings** — 3-5 bullet points summarizing the most important results.
2. **Representative Papers** — A markdown table with columns: Paper | Year | Core Idea | Evidence.
3. **Method Landscape** — Categorize the main approaches and describe each.
4. **Experimental Comparison** — Compare datasets, metrics, and results where applicable.
5. **Limitations & Open Questions** — Note gaps, conflicting claims, and areas for future work.
6. **Further Reading** — Suggest related topics.

Critical rules:
- Every factual claim MUST cite a paper by title and year.
- If evidence is thin or missing for a claim, mark it with ⚠️ and note the uncertainty.
- Do NOT fabricate results — only use the provided paper information.
- Use academic English but keep it readable."""


async def synthesize(state: AgentState) -> dict:
    """Draft a review/survey from the collected evidence and claims.

    Uses LLM to generate a structured survey in Markdown format.
    On LLM failure, falls back to deterministic `render_review_markdown()`.
    """
    now = datetime.now(timezone.utc).isoformat()
    errors: list[str] = []
    fallback_used = False

    try:
        llm = get_llm()

        if not llm.available:
            logger.info("LLM not available, using template-based synthesis")
            fallback_used = True
            draft = render_review_markdown(state)
        else:
            prompt = _build_synthesis_prompt(state)
            draft = await llm.generate(
                system=SYNTHESIS_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4096,
                temperature=0.3,
            )
            if not draft.strip():
                logger.warning("Synthesizer returned empty response, falling back")
                fallback_used = True
                draft = render_review_markdown(state)
    except Exception as e:
        logger.error("Synthesizer LLM call failed, using template fallback: %s", e)
        errors.append(f"Synthesizer failed: {e}")
        draft = render_review_markdown(state)
        fallback_used = True

    return {
        "review_draft": draft,
        "errors": errors,
        "trace": [
            TraceEvent(
                timestamp=now,
                step="synthesize",
                event_type="end",
                data={
                    "claims_used": len(state.get("claims", [])),
                    "papers_cited": len(state.get("ranked_papers", [])),
                    "fallback_used": fallback_used,
                },
            )
        ],
    }


def _build_synthesis_prompt(state: AgentState) -> str:
    """Build a detailed prompt for the LLM synthesizer from available state."""
    parts: list[str] = []
    query = state.get("query", "Unknown Topic")
    ranked = state.get("ranked_papers", [])
    claims = state.get("claims", [])
    comparison = state.get("comparison")

    parts.append(f"**Research Topic**: {query}")
    parts.append("")

    if ranked:
        parts.append("## Papers Retrieved")
        parts.append("")
        for i, p in enumerate(ranked, 1):
            parts.append(f"### {i}. {p.title}")
            parts.append(f"- Authors: {', '.join(p.authors[:5])}")
            parts.append(f"- Year: {p.year or 'N/A'}")
            parts.append(f"- Venue: {p.venue or 'N/A'}")
            citation_str = 'N/A' if p.citation_count is None else str(p.citation_count)
            parts.append(f"- Citations: {citation_str}")
            parts.append(f"- URL: {p.url}")
            if p.abstract:
                parts.append(f"- Abstract: {p.abstract[:500]}")
            parts.append("")
    else:
        parts.append("*(No papers retrieved — generate a placeholder survey.)*")
        parts.append("")

    if claims:
        parts.append("## Extracted Claims")
        parts.append("")
        paper_claims: dict[str, list] = {}
        for c in claims:
            paper_claims.setdefault(c.paper_id, []).append(c)
        for pid, pclaims in paper_claims.items():
            parts.append(f"### {pid}")
            for c in pclaims:
                parts.append(f"- [{c.claim_type}] {c.claim_text}")
                parts.append(f"  Evidence: {c.evidence[:200]}")
                parts.append(f"  Section: {c.section}, Page: {c.page or 'N/A'}")
            parts.append("")
    else:
        parts.append("*(No claims extracted — use only paper metadata above.)*")
        parts.append("")

    if comparison and comparison.rows:
        parts.append("## Comparison Data")
        parts.append("")
        parts.append(f"Dimensions: {', '.join(comparison.dimensions)}")
        if comparison.caveats:
            parts.append(f"Caveats: {comparison.caveats}")
        parts.append("")
    elif ranked and len(ranked) >= 2:
        parts.append("*(Comparison data not available — note this in the survey.)*")
        parts.append("")

    parts.append("---")
    parts.append("Write the structured literature survey following the system prompt structure.")
    parts.append("Cite papers as: [Title] (Year).")

    return "\n".join(parts)
