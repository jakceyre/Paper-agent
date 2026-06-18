"""Tests for the ranker agent."""

import pytest
from paper_agent.agents.ranker import rank, _deduplicate_by_title
from paper_agent.models.paper import PaperMetadata
from paper_agent.state import initial_state


@pytest.mark.asyncio
async def test_rank_empty_results():
    state = initial_state("test")
    result = await rank(state)
    assert result["ranked_papers"] == []
    assert result["status"] == "no_results"


@pytest.mark.asyncio
async def test_rank_respects_max_papers():
    state = initial_state("test", max_papers=3)
    papers = [
        PaperMetadata(paper_id=f"paper_{i}", source="arxiv", title=f"Test Paper {i}", authors=["Author"], year=2023)
        for i in range(10)
    ]
    state["search_results"] = papers
    result = await rank(state)
    assert len(result["ranked_papers"]) <= 3


def test_deduplicate_by_title_removes_duplicates():
    papers = [
        PaperMetadata(paper_id="id1", source="arxiv", title="Same Title", authors=["A"], year=2023),
        PaperMetadata(paper_id="id2", source="semantic_scholar", title="Same Title", authors=["A"], year=2023),
        PaperMetadata(paper_id="id3", source="arxiv", title="Different Title", authors=["B"], year=2024),
    ]
    result = _deduplicate_by_title(papers)
    assert len(result) == 2
