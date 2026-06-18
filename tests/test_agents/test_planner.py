"""Tests for the planner agent."""

from unittest.mock import AsyncMock, patch

import pytest
from paper_agent.agents.planner import plan, _parse_queries
from paper_agent.state import initial_state


def test_parse_queries_q_format():
    text = "Q: retrieval augmented generation\nQ: multi-hop reasoning RAG"
    result = _parse_queries(text)
    assert result == ["retrieval augmented generation", "multi-hop reasoning RAG"]


def test_parse_queries_numbered():
    text = "1. Retrieval Augmented Generation\n2. Multi-hop QA survey"
    result = _parse_queries(text)
    assert len(result) == 2
    assert "Retrieval Augmented Generation" in result


def test_parse_queries_bullets():
    text = "- retrieval augmented generation\n- multi-hop reasoning"
    result = _parse_queries(text)
    assert len(result) == 2


def test_parse_queries_deduplicates():
    text = "Q: RAG systems\nQ: rag systems\nQ: RAG Systems"
    result = _parse_queries(text)
    assert len(result) == 1


def test_parse_queries_short_lines_filtered():
    text = "Q: ab\nQ: RAG systems"
    result = _parse_queries(text)
    assert result == ["RAG systems"]


def test_parse_queries_empty_input():
    assert _parse_queries("") == []
    assert _parse_queries("   \n\n  ") == []


@pytest.mark.asyncio
async def test_plan_creates_search_queries():
    state = initial_state("multi-modal retrieval")
    result = await plan(state)
    assert "search_queries" in result
    assert len(result["search_queries"]) >= 1
    assert result["search_queries"][0] == "multi-modal retrieval"


@pytest.mark.asyncio
async def test_plan_includes_trace():
    state = initial_state("test")
    result = await plan(state)
    assert "trace" in result
    assert result["trace"][0].step == "plan"
    assert result["trace"][0].event_type == "end"


@pytest.mark.asyncio
async def test_plan_includes_errors_key():
    state = initial_state("test")
    result = await plan(state)
    assert "errors" in result
    assert isinstance(result["errors"], list)


@pytest.mark.asyncio
async def test_plan_calls_llm_and_parses_queries():
    state = initial_state("RAG for multi-hop QA")
    mock_llm = AsyncMock()
    mock_llm.available = True
    mock_llm.generate.return_value = "Q: retrieval augmented generation\nQ: multi-hop reasoning RAG"

    with patch("paper_agent.agents.planner.get_llm", return_value=mock_llm):
        result = await plan(state)

    assert mock_llm.generate.called
    assert len(result["search_queries"]) == 2
    assert "retrieval augmented generation" in result["search_queries"]


@pytest.mark.asyncio
async def test_plan_fallback_on_llm_error():
    state = initial_state("RAG survey")
    mock_llm = AsyncMock()
    mock_llm.available = True
    mock_llm.generate.side_effect = RuntimeError("API error")

    with patch("paper_agent.agents.planner.get_llm", return_value=mock_llm):
        result = await plan(state)

    assert result["search_queries"] == ["RAG survey"]
    assert len(result["errors"]) >= 1


@pytest.mark.asyncio
async def test_plan_passes_llm_disabled():
    state = initial_state("RAG survey")
    mock_llm = AsyncMock()
    mock_llm.available = False

    with patch("paper_agent.agents.planner.get_llm", return_value=mock_llm):
        result = await plan(state)

    assert result["search_queries"] == ["RAG survey"]
    assert result["errors"] == []
