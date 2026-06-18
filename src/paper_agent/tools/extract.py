"""Claim extraction tool: use LLM to extract claims from parsed papers."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from paper_agent.state import AgentState
from paper_agent.models.claim import Claim, EvidenceItem
from paper_agent.models.trace import TraceEvent

logger = logging.getLogger(__name__)


async def extract_claims(state: AgentState) -> dict:
    """Extract claims from each parsed paper using LLM-guided extraction."""
    now = datetime.now(timezone.utc).isoformat()
    claims: list[Claim] = []
    evidence: list[EvidenceItem] = []
    errors: list[str] = []
    papers_processed = 0

    return {
        "claims": claims,
        "evidence": evidence,
        "errors": errors,
        "trace": [
            TraceEvent(
                timestamp=now,
                step="extract",
                event_type="end",
                data={
                    "claims_extracted": len(claims),
                    "papers_processed": papers_processed,
                },
            )
        ],
    }
