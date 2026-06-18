"""Shared test fixtures for Paper Agent."""

from __future__ import annotations

import pytest
from paper_agent.state import initial_state, AgentState
from paper_agent.models.paper import PaperMetadata, PaperSection, ParsedPaper
from paper_agent.models.claim import Claim, EvidenceItem


@pytest.fixture
def sample_query() -> str:
    return "Retrieval Augmented Generation for multi-hop QA"


@pytest.fixture
def sample_state(sample_query: str) -> AgentState:
    """Create a minimal initial state for testing."""
    return initial_state(sample_query)


@pytest.fixture
def sample_paper() -> PaperMetadata:
    """Create a sample paper metadata for testing."""
    return PaperMetadata(
        paper_id="2301.12345",
        source="arxiv",
        title="RAG for Multi-Hop Question Answering",
        authors=["Alice Chen", "Bob Wang"],
        year=2023,
        abstract="We propose a novel RAG approach for multi-hop QA...",
        url="https://arxiv.org/abs/2301.12345",
        pdf_url="https://arxiv.org/pdf/2301.12345",
        citation_count=42,
        venue="ACL 2023",
    )


@pytest.fixture
def sample_parsed_paper(sample_paper: PaperMetadata) -> ParsedPaper:
    """Create a sample parsed paper for testing."""
    return ParsedPaper(
        paper_id="2301.12345",
        metadata=sample_paper,
        sections=[
            PaperSection(
                heading="Abstract",
                level=1,
                page_start=1,
                page_end=1,
                text="We propose a novel RAG approach for multi-hop question answering...",
            ),
            PaperSection(
                heading="3. Method",
                level=1,
                page_start=3,
                page_end=5,
                text="Our method consists of three stages: retrieval, reranking, and generation...",
            ),
        ],
        full_text="We propose a novel RAG approach for multi-hop QA...\n\nOur method consists of three stages...",
        page_count=8,
    )


@pytest.fixture
def sample_claim() -> Claim:
    """Create a sample claim for testing."""
    return Claim(
        paper_id="2301.12345",
        claim_type="method",
        claim_text="Three-stage pipeline: retrieval, reranking, generation",
        evidence="Our method consists of three stages: retrieval, reranking, and generation...",
        section="3. Method",
        page=3,
        confidence=0.9,
    )


@pytest.fixture
def sample_evidence(sample_claim: Claim) -> EvidenceItem:
    """Create a sample evidence item for testing."""
    return EvidenceItem(
        evidence_id="ev_001",
        paper_id="2301.12345",
        claim_text=sample_claim.claim_text,
        quote=sample_claim.evidence,
        section=sample_claim.section,
        page=sample_claim.page,
        url="https://arxiv.org/abs/2301.12345",
        timestamp="2025-01-01T00:00:00Z",
    )
