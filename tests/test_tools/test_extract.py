"""Tests for claim extraction tool."""

from unittest.mock import AsyncMock, patch

import pytest
from paper_agent.tools.extract import extract_claims, _build_chunks
from paper_agent.state import initial_state


# ── _build_chunks tests ────────────────────────────────


def test_build_chunks_from_sections():
    """Chunks should be built from sections when available."""

    class FakeSection:
        def __init__(self, heading, text, page_start):
            self.heading = heading
            self.text = text
            self.page_start = page_start

    sections = [
        FakeSection("1. Intro", "Introduction text here.", 1),
        FakeSection("2. Method", "We propose a novel method.", 3),
    ]
    chunks = _build_chunks(sections, "", 500)
    assert len(chunks) == 2
    assert chunks[0][1] == "1. Intro"
    assert chunks[0][2] == 1
    assert chunks[1][1] == "2. Method"
    assert chunks[1][2] == 3


def test_build_chunks_splits_long_sections():
    """Long sections should be split at chunk_size boundaries."""

    class FakeSection:
        def __init__(self, text):
            self.heading = "1. Intro"
            self.text = text
            self.page_start = 1

    long_text = "x" * 2500
    sections = [FakeSection(long_text)]
    chunks = _build_chunks(sections, "", 1000)
    assert len(chunks) >= 2


def test_build_chunks_fallback():
    """Without sections, fall back to full_text chunking."""
    text = "a" * 2500
    chunks = _build_chunks([], text, 1000)
    assert len(chunks) >= 2
    assert chunks[0][1] == "Full Text"  # default heading


def test_build_chunks_empty():
    """Empty sections and text should return empty list."""
    assert _build_chunks([], "", 1000) == []


# ── extract_claims tests ───────────────────────────────


@pytest.mark.asyncio
async def test_extract_claims_returns_expected_keys():
    """extract_claims should return claims, evidence, errors, trace."""
    state = initial_state("test")
    result = await extract_claims(state)
    assert "claims" in result
    assert "evidence" in result
    assert "errors" in result
    assert "trace" in result
    assert isinstance(result["claims"], list)


@pytest.mark.asyncio
async def test_extract_claims_empty_when_no_parsed():
    """Without parsed papers, returns empty claims."""
    state = initial_state("test")
    result = await extract_claims(state)
    assert result["claims"] == []
    assert result["evidence"] == []


@pytest.mark.asyncio
async def test_extract_claims_with_mock_llm():
    """Claims should be extracted when LLM returns valid JSON."""

    class FakeSection:
        def __init__(self, heading, text, page_start):
            self.heading = heading
            self.text = text
            self.page_start = page_start

    sections = [
        FakeSection("3. Method", "We propose a novel RAG architecture.", 3),
    ]

    class FakeParsedPaper:
        def __init__(self):
            self.sections = sections
            self.full_text = "We propose a novel RAG architecture."
            self.metadata = type(
                "obj", (object,),
                {"pdf_url": "https://arxiv.org/pdf/2301.12345", "url": "https://arxiv.org/abs/2301.12345"},
            )()

    state = initial_state("test")
    state["parsed_papers"] = {"2301.12345": FakeParsedPaper()}

    mock_llm = AsyncMock()
    mock_llm.generate_with_json.return_value = {
        "claims": [
            {
                "claim_type": "method",
                "claim_text": "Three-stage RAG pipeline",
                "evidence": "We propose a novel RAG architecture.",
                "confidence": 0.9,
            }
        ]
    }

    with patch("paper_agent.tools.extract.get_llm", return_value=mock_llm):
        result = await extract_claims(state)

    assert len(result["claims"]) == 1
    assert result["claims"][0].claim_type == "method"
    assert result["claims"][0].paper_id == "2301.12345"
    assert result["claims"][0].section == "3. Method"
    assert result["claims"][0].page == 3
    assert len(result["evidence"]) == 1
    assert result["evidence"][0].paper_id == "2301.12345"


@pytest.mark.asyncio
async def test_extract_claims_handles_parse_error():
    """When LLM returns a parse error, it should be recorded and skipped."""

    class FakeSection:
        def __init__(self):
            self.heading = "Abstract"
            self.text = "test text"
            self.page_start = 1

    class FakeParsedPaper:
        def __init__(self):
            self.sections = [FakeSection()]
            self.full_text = "test text"
            self.metadata = type("obj", (object,), {"pdf_url": None, "url": ""})()

    state = initial_state("test")
    state["parsed_papers"] = {"bad_paper": FakeParsedPaper()}

    mock_llm = AsyncMock()
    mock_llm.generate_with_json.return_value = {
        "_parse_error": True,
        "_raw_text": "not json",
        "_reason": "no JSON object found",
    }

    with patch("paper_agent.tools.extract.get_llm", return_value=mock_llm):
        result = await extract_claims(state)

    assert result["claims"] == []
    assert len(result["errors"]) >= 1
    assert "JSON parse error" in result["errors"][0]


@pytest.mark.asyncio
async def test_extract_claims_handles_llm_exception():
    """When LLM raises an exception, it should be captured."""

    class FakeSection:
        def __init__(self):
            self.heading = "Abstract"
            self.text = "test text"
            self.page_start = 1

    class FakeParsedPaper:
        def __init__(self):
            self.sections = [FakeSection()]
            self.full_text = "test text"
            self.metadata = type("obj", (object,), {"pdf_url": None, "url": ""})()

    state = initial_state("test")
    state["parsed_papers"] = {"bad_paper": FakeParsedPaper()}

    mock_llm = AsyncMock()
    mock_llm.generate_with_json.side_effect = RuntimeError("API error")

    with patch("paper_agent.tools.extract.get_llm", return_value=mock_llm):
        result = await extract_claims(state)

    assert result["claims"] == []
    assert len(result["errors"]) >= 1
