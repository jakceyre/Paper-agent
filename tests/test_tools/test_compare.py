"""Tests for paper comparison tool."""

from unittest.mock import AsyncMock, patch

import pytest
from paper_agent.tools.compare import compare_papers, _stub_comparison
from paper_agent.models.claim import Claim
from paper_agent.state import initial_state


def test_stub_comparison():
    """Stub should produce a table with correct paper IDs."""
    table = _stub_comparison(["p1", "p2"])
    assert table.paper_ids == ["p1", "p2"]
    assert len(table.dimensions) >= 2
    assert "LLM" in table.caveats


@pytest.mark.asyncio
async def test_compare_skips_when_less_than_two_papers():
    """Should skip comparison when fewer than 2 papers have claims."""
    state = initial_state("test")
    # Only 1 paper with claims
    state["claims"] = [
        Claim(
            paper_id="paper_1",
            claim_type="method",
            claim_text="A method",
            evidence="evidence text",
            section="3. Method",
        )
    ]
    result = await compare_papers(state)
    assert result["comparison"] is None
    assert result["trace"][0].event_type == "info"


@pytest.mark.asyncio
async def test_compare_with_mock_llm():
    """Should produce comparison table when LLM returns valid JSON."""
    state = initial_state("test")
    state["claims"] = [
        Claim(paper_id="p1", claim_type="method", claim_text="BERT method", evidence="We use BERT", section="M"),
        Claim(paper_id="p2", claim_type="method", claim_text="T5 method", evidence="We use T5", section="M"),
        Claim(paper_id="p1", claim_type="dataset", claim_text="SQuAD", evidence="on SQuAD", section="E"),
        Claim(paper_id="p2", claim_type="dataset", claim_text="NaturalQ", evidence="on NQ", section="E"),
    ]

    mock_llm = AsyncMock()
    mock_llm.available = True
    mock_llm.generate_with_json.return_value = {
        "dimensions": ["Method", "Dataset"],
        "rows": [
            {"dimension": "Method", "paper_id": "p1", "value": "BERT-based"},
            {"dimension": "Method", "paper_id": "p2", "value": "T5-based"},
            {"dimension": "Dataset", "paper_id": "p1", "value": "SQuAD"},
            {"dimension": "Dataset", "paper_id": "p2", "value": "Natural Questions"},
        ],
        "caveats": "Different evaluation datasets.",
    }

    with patch("paper_agent.tools.compare.get_llm", return_value=mock_llm):
        result = await compare_papers(state)

    assert result["comparison"] is not None
    assert len(result["comparison"].rows) == 4
    assert result["comparison"].caveats == "Different evaluation datasets."


@pytest.mark.asyncio
async def test_compare_fallback_on_parse_error():
    """Should use stub when JSON parsing fails."""
    state = initial_state("test")
    state["claims"] = [
        Claim(paper_id="p1", claim_type="method", claim_text="M1", evidence="e", section="S"),
        Claim(paper_id="p2", claim_type="method", claim_text="M2", evidence="e", section="S"),
    ]

    mock_llm = AsyncMock()
    mock_llm.available = True
    mock_llm.generate_with_json.return_value = {
        "_parse_error": True,
        "_reason": "no JSON found",
    }

    with patch("paper_agent.tools.compare.get_llm", return_value=mock_llm):
        result = await compare_papers(state)

    assert result["comparison"] is not None
    assert "LLM" in result["comparison"].caveats  # stub used


@pytest.mark.asyncio
async def test_compare_fallback_on_exception():
    """Should use stub when LLM raises exception."""
    state = initial_state("test")
    state["claims"] = [
        Claim(paper_id="p1", claim_type="method", claim_text="M1", evidence="e", section="S"),
        Claim(paper_id="p2", claim_type="method", claim_text="M2", evidence="e", section="S"),
    ]

    mock_llm = AsyncMock()
    mock_llm.available = True
    mock_llm.generate_with_json.side_effect = RuntimeError("API error")

    with patch("paper_agent.tools.compare.get_llm", return_value=mock_llm):
        result = await compare_papers(state)

    assert len(result["errors"]) >= 1
    assert result["comparison"] is not None  # stub fallback
