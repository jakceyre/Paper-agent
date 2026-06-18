"""Tests for paper search tools."""

from unittest.mock import AsyncMock, patch

import pytest
from paper_agent.tools.search import search_papers, _arxiv_id_from_entry
from paper_agent.models.paper import PaperMetadata
from paper_agent.state import initial_state


def test_arxiv_id_from_entry_with_version():
    assert _arxiv_id_from_entry("http://arxiv.org/abs/2406.18007v1") == "2406.18007"
    assert _arxiv_id_from_entry("http://arxiv.org/abs/2301.12345v9") == "2301.12345"


def test_arxiv_id_from_entry_without_version():
    assert _arxiv_id_from_entry("http://arxiv.org/abs/2301.12345") == "2301.12345"


def test_arxiv_id_from_entry_trailing_slash():
    assert _arxiv_id_from_entry("http://arxiv.org/abs/2406.18007/") == "2406.18007"


@pytest.mark.asyncio
async def test_search_papers_returns_expected_keys():
    state = initial_state("test query")
    result = await search_papers(state)
    assert "search_results" in result
    assert "trace" in result
    assert "errors" in result
    assert isinstance(result["search_results"], list)
    assert len(result["trace"]) > 0


@pytest.mark.asyncio
async def test_search_papers_empty_without_queries():
    state = initial_state("")
    result = await search_papers(state)
    assert isinstance(result["search_results"], list)
    assert "errors" in result


@pytest.mark.asyncio
async def test_search_papers_structure_with_mock():
    state = initial_state("RAG multi-hop QA")
    state["search_queries"] = ["RAG multi-hop QA"]

    with patch(
        "paper_agent.tools.search._search_arxiv",
        AsyncMock(return_value=[
            PaperMetadata(paper_id="2301.00001", source="arxiv", title="RAG for Multi-Hop QA",
                          authors=["Alice Chen"], year=2023, abstract="We propose a RAG approach...",
                          url="https://arxiv.org/abs/2301.00001", pdf_url="https://arxiv.org/pdf/2301.00001")
        ]),
    ), patch(
        "paper_agent.tools.search._search_s2",
        AsyncMock(return_value=[
            PaperMetadata(paper_id="2301.00002", source="semantic_scholar", title="Multi-Hop Retrieval Survey",
                          authors=["Bob Wang"], year=2023, abstract="A survey of multi-hop retrieval...",
                          url="https://api.semanticscholar.org/paper/2301.00002")
        ]),
    ):
        result = await search_papers(state)

    assert len(result["search_results"]) == 2
    sources = {p.source for p in result["search_results"]}
    assert "arxiv" in sources
    assert "semantic_scholar" in sources
    assert len(result["errors"]) == 0


@pytest.mark.asyncio
async def test_search_papers_handles_api_failure():
    state = initial_state("test")
    state["search_queries"] = ["test"]

    with patch(
        "paper_agent.tools.search._search_arxiv",
        AsyncMock(side_effect=RuntimeError("arXiv down")),
    ), patch(
        "paper_agent.tools.search._search_s2",
        AsyncMock(return_value=[
            PaperMetadata(paper_id="s2_only", source="semantic_scholar", title="S2 Only Paper",
                          authors=["Author"], year=2024)
        ]),
    ):
        result = await search_papers(state)

    assert len(result["search_results"]) == 1
    assert result["search_results"][0].source == "semantic_scholar"
    assert len(result["errors"]) >= 1
    assert any("arxiv" in e.lower() for e in result["errors"])


@pytest.mark.asyncio
async def test_search_papers_deduplicates_by_paper_id():
    state = initial_state("test")
    state["search_queries"] = ["test"]

    paper = PaperMetadata(paper_id="2301.12345", source="arxiv", title="Same Paper", authors=["Author"], year=2023)

    with patch("paper_agent.tools.search._search_arxiv", AsyncMock(return_value=[paper])), patch(
        "paper_agent.tools.search._search_s2",
        AsyncMock(return_value=[
            PaperMetadata(paper_id="2301.12345", source="semantic_scholar", title="Same Paper",
                          authors=["Author"], year=2023)
        ]),
    ):
        result = await search_papers(state)

    assert len(result["search_results"]) == 1


@pytest.mark.asyncio
async def test_search_papers_respects_max_papers():
    state = initial_state("test", max_papers=3)
    state["search_queries"] = ["test"]

    papers = [
        PaperMetadata(paper_id=f"paper_{i}", source="arxiv", title=f"Paper {i}", authors=["A"], year=2023)
        for i in range(10)
    ]

    with patch("paper_agent.tools.search._search_arxiv", AsyncMock(return_value=papers)), patch(
        "paper_agent.tools.search._search_s2", AsyncMock(return_value=[])
    ):
        result = await search_papers(state)

    assert len(result["search_results"]) <= 3
