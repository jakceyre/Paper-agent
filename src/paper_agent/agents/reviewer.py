"""Reviewer agent: critiques and improves the synthesis draft."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from paper_agent.state import AgentState
from paper_agent.models.trace import TraceEvent

logger = logging.getLogger(__name__)


async def review(state: AgentState) -> dict:
    """Critique the review draft and produce a polished final version.

    Checks for:
    1. Unsupported claims (claims without evidence).
    2. Missing sections (method, experiments, limitations).
    3. Citation accuracy.
    4. Uncertainty language where evidence is thin.

    For MVP skeleton: passes the draft through unchanged.
    Error handling: any LLM failure falls back to original draft.
    """
    now = datetime.now(timezone.utc).isoformat()
    draft = state.get("review_draft", "")
    errors: list[str] = []
    changes_made = False

    try:
        # TODO: Implement LLM-based review
        final = draft  # Stub: no changes
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
                    "fallback_used": len(errors) > 0,
                },
            )
        ],
    }
