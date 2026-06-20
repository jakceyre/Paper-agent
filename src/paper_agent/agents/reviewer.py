"""Reviewer agent: critiques and improves the synthesis draft.

Uses LLM to check the review for unsupported claims, missing sections,
citation accuracy, and uncertainty language.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from paper_agent.llm.client import get_llm
from paper_agent.state import AgentState
from paper_agent.models.trace import TraceEvent

logger = logging.getLogger(__name__)

REVIEWER_SYSTEM = """\
You are an academic review editor. Critique and improve a literature survey draft.

Check for:
1. **Unsupported claims**: Any factual statement without a cited paper as evidence.
2. **Missing sections**: Method landscape, experimental comparison, or limitations absent.
3. **Citation accuracy**: Claims that don't match what the cited paper actually says.
4. **Uncertainty language**: Where evidence is thin, add "preliminary evidence suggests" or similar hedging.

Rules:
- If the survey already has good evidence coverage, make minimal changes.
- Add ⚠️ markers for claims needing verification.
- Preserve the original Markdown structure (headings, table format).
- Do NOT fabricate new papers or claims — only use information already in the draft.

Output: the improved Markdown survey text (no JSON wrapper, just the Markdown)."""


async def review(state: AgentState) -> dict:
    """Critique the review draft and produce a polished final version.

    Checks for unsupported claims, missing sections, citation accuracy,
    and appropriate uncertainty language.

    On LLM failure, passes the draft through unchanged.
    """
    now = datetime.now(timezone.utc).isoformat()
    draft = state.get("review_draft", "")
    claims_count = len(state.get("claims", []))
    papers_count = len(state.get("ranked_papers", []))
    errors: list[str] = []
    changes_made = False

    try:
        llm = get_llm()

        if not llm.available:
            logger.info("LLM not available for review, passing draft through")
            final = draft
        else:
            # Build critique prompt with context
            prompt = (
                f"The survey has {papers_count} papers cited and {claims_count} "
                f"extracted claims.\n\n--- DRAFT ---\n{draft}\n---\n\n"
                f"Review and improve this survey following the system instructions. "
                f"Return the full improved Markdown."
            )
            reviewed = await llm.generate(
                system=REVIEWER_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4096,
                temperature=0.3,
            )

            if reviewed.strip() and len(reviewed) > len(draft) * 0.5:
                # Ensure the output is valid — at least half the original length
                final = reviewed.strip()
                changes_made = final != draft
            else:
                logger.warning("Reviewer returned too-short response, keeping original")
                final = draft
    except Exception as e:
        logger.error("Reviewer LLM call failed, using original draft: %s", e)
        errors.append(f"Reviewer failed: {e}")
        final = draft

    return {
        "final_review": final,
        "status": "complete",
        "errors": errors,
        "trace": [
            TraceEvent(
                timestamp=now,
                step="review_critique",
                event_type="end",
                data={
                    "changes_made": changes_made,
                    "fallback_used": len(errors) > 0 or not changes_made,
                },
            )
        ],
    }
