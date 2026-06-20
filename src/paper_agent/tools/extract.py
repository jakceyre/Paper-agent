"""Claim extraction tool: use LLM to extract claims from parsed papers.

Each claim carries mandatory evidence grounding: paper_id, section, and page number.
The LLM is called per-chunk with structured JSON output expectations.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from paper_agent.config import load_config
from paper_agent.llm.client import get_llm
from paper_agent.state import AgentState
from paper_agent.models.claim import Claim, EvidenceItem
from paper_agent.models.trace import TraceEvent

logger = logging.getLogger(__name__)

EXTRACTION_SYSTEM = """\
You are an academic paper information extraction specialist. Given a text \
chunk from a research paper, extract all factual claims.

For each claim, provide:
- claim_type: one of "contribution", "method", "dataset", "metric", "limitation"
- claim_text: a concise statement of the claim (1 sentence)
- evidence: a direct quote from the text that supports this claim (max 200 chars)
- confidence: your confidence that this claim is accurate (0.0 - 1.0)

Important:
- Only extract claims that are explicitly stated in the text.
- Do NOT fabricate or infer claims not present.
- If no clear claims exist, return an empty claims list.
- Each piece of evidence must be a verbatim quote from the provided text.

Output as JSON:
{"claims": [{"claim_type": "method", "claim_text": "...", "evidence": "...", "confidence": 0.9}]}"""


async def extract_claims(state: AgentState) -> dict:
    """Extract claims from each parsed paper using LLM-guided extraction.

    For each paper in state.parsed_papers:
      1. Chunk the full_text into section-based segments of ~parse_chunk_size chars.
      2. For each chunk, prompt the LLM to extract claims.
      3. Each claim includes paper_id, section heading, and page number.
      4. Also produces an EvidenceItem for the evidence store.

    Per-paper and per-chunk errors are captured without aborting the entire run.

    Returns partial state dict with:
        - claims: list of Claim objects (Annotated[list, add])
        - evidence: list of EvidenceItem objects (Annotated[list, add])
        - errors: per-chunk failures
        - trace: extraction events
    """
    now_ts = datetime.now(timezone.utc).isoformat()
    config = load_config()
    parsed_papers = state.get("parsed_papers", {})
    chunk_size = config.pdf.parse_chunk_size

    claims: list[Claim] = []
    evidence_items: list[EvidenceItem] = []
    errors: list[str] = []
    papers_processed = 0
    chunks_processed = 0

    llm = get_llm()

    for paper_id, paper in parsed_papers.items():
        try:
            # Build chunks from sections (preserving section boundaries)
            chunks = _build_chunks(paper.sections, paper.full_text, chunk_size)
            if not chunks:
                continue

            paper_claims = 0
            for chunk_idx, (chunk_text, section, page) in enumerate(chunks):
                if not chunk_text.strip():
                    continue

                try:
                    # Build prompt with section and page context
                    prompt = (
                        f"Section: {section}\n"
                        f"Page: {page}\n\n"
                        f"Text:\n{chunk_text}"
                    )
                    result = await llm.generate_with_json(
                        system=EXTRACTION_SYSTEM,
                        prompt=prompt,
                        max_tokens=2048,
                    )

                    if result.get("_parse_error"):
                        errors.append(
                            f"JSON parse error for {paper_id} chunk {chunk_idx}: "
                            f"{result.get('_reason', 'unknown')}"
                        )
                        continue

                    for c in result.get("claims", []):
                        claim = Claim(
                            paper_id=paper_id,
                            claim_type=c.get("claim_type", "method"),
                            claim_text=c.get("claim_text", ""),
                            evidence=c.get("evidence", ""),
                            section=section,
                            page=page,
                            confidence=float(c.get("confidence", 0.8)),
                        )
                        claims.append(claim)
                        paper_claims += 1

                        # Also create evidence item
                        evidence_items.append(
                            EvidenceItem(
                                evidence_id=uuid.uuid4().hex[:12],
                                paper_id=paper_id,
                                claim_text=claim.claim_text,
                                quote=claim.evidence,
                                section=section,
                                page=page,
                                url=paper.metadata.pdf_url or paper.metadata.url,
                                timestamp=now_ts,
                            )
                        )
                    chunks_processed += 1

                except Exception as e:
                    errors.append(
                        f"Extraction failed for {paper_id} chunk {chunk_idx}: {e}"
                    )
                    continue

            papers_processed += 1

        except Exception as e:
            msg = f"Extraction failed for {paper_id}: {e}"
            logger.error(msg)
            errors.append(msg)

    return {
        "claims": claims,
        "evidence": evidence_items,
        "errors": errors,
        "trace": [
            TraceEvent(
                timestamp=now_ts,
                step="extract",
                event_type="end",
                data={
                    "claims_extracted": len(claims),
                    "evidence_items": len(evidence_items),
                    "papers_processed": papers_processed,
                    "chunks_processed": chunks_processed,
                },
            )
        ],
    }


def _build_chunks(
    sections: list,
    full_text: str,
    chunk_size: int,
) -> list[tuple[str, str, int]]:
    """Split a paper's text into chunks for LLM processing.

    Prefers section-boundary chunking. Falls back to fixed-size chunking
    on full_text if sections are insufficient.

    Returns:
        List of (chunk_text, section_heading, page_number) tuples.
    """
    chunks: list[tuple[str, str, int]] = []

    if sections:
        for sec in sections:
            text = sec.text if hasattr(sec, "text") else ""
            heading = sec.heading if hasattr(sec, "heading") else ""
            page = sec.page_start if hasattr(sec, "page_start") else 1
            if not text.strip():
                continue
            if len(text) <= chunk_size:
                chunks.append((text, heading, page))
            else:
                # Split long sections into sub-chunks
                for i in range(0, len(text), chunk_size):
                    sub = text[i:i + chunk_size]
                    if sub.strip():
                        chunks.append((sub, heading, page))
    else:
        # Fallback: chunk by fixed size
        for i in range(0, len(full_text), chunk_size):
            sub = full_text[i:i + chunk_size]
            if sub.strip():
                chunks.append((sub, "Full Text", 1))

    return chunks
