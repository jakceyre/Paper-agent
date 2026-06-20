"""Tests for the reviewer agent."""

from unittest.mock import AsyncMock, patch

import pytest
from paper_agent.agents.reviewer import review
from paper_agent.state import initial_state


@pytest.mark.asyncio
async def test_review_returns_expected_keys():
    """Reviewer should return final_review, status, errors, trace."""
    state = initial_state("test")
    state["review_draft"] = "# Test Review\nSome content."
    result = await review(state)

    assert "final_review" in result
    assert "status" in result
    assert "errors" in result
    assert "trace" in result
    assert result["status"] == "complete"


@pytest.mark.asyncio
async def test_review_passes_through_without_llm():
    """Without LLM, draft should pass through unchanged."""
    state = initial_state("test")
    draft = "# Survey\nTest content."
    state["review_draft"] = draft
    result = await review(state)

    assert result["final_review"] == draft


@pytest.mark.asyncio
async def test_review_with_mock_llm():
    """LLM review should produce improved text."""
    state = initial_state("test")
    state["review_draft"] = "# Survey\nOriginal draft."
    state["claims"] = []
    state["ranked_papers"] = []

    mock_llm = AsyncMock()
    mock_llm.available = True
    mock_llm.generate.return_value = "# Survey\nImproved with review notes."

    with patch("paper_agent.agents.reviewer.get_llm", return_value=mock_llm):
        result = await review(state)

    assert "Improved" in result["final_review"]


@pytest.mark.asyncio
async def test_review_rejects_too_short_response():
    """If reviewed text is much shorter than original, keep original."""
    state = initial_state("test")
    state["review_draft"] = "# Survey\n" + "x" * 500
    state["claims"] = []
    state["ranked_papers"] = []

    mock_llm = AsyncMock()
    mock_llm.available = True
    mock_llm.generate.return_value = "Too short."

    with patch("paper_agent.agents.reviewer.get_llm", return_value=mock_llm):
        result = await review(state)

    # Should keep original since "Too short." < 250 chars
    assert result["final_review"] == state["review_draft"]


@pytest.mark.asyncio
async def test_review_fallback_on_error():
    """On LLM error, draft should pass through with error recorded."""
    state = initial_state("test")
    state["review_draft"] = "# Survey\nOriginal."

    mock_llm = AsyncMock()
    mock_llm.available = True
    mock_llm.generate.side_effect = RuntimeError("API error")

    with patch("paper_agent.agents.reviewer.get_llm", return_value=mock_llm):
        result = await review(state)

    assert result["final_review"] == "# Survey\nOriginal."
    assert len(result["errors"]) >= 1
