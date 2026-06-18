"""Claim and evidence models — the core evidence-grounded output units."""

from __future__ import annotations

from pydantic import BaseModel


class Claim(BaseModel):
    """A single claim extracted from a paper, with mandatory evidence grounding."""

    paper_id: str
    claim_type: str
    claim_text: str
    evidence: str
    section: str
    page: int | None = None
    confidence: float = 1.0


class EvidenceItem(BaseModel):
    """A discrete piece of evidence stored in the Evidence Store."""

    evidence_id: str
    paper_id: str
    claim_text: str
    quote: str | None = None
    section: str = ""
    page: int | None = None
    url: str = ""
    timestamp: str = ""
