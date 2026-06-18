"""Integration test: full graph execution with mocked search to avoid API calls."""

from unittest.mock import AsyncMock, patch

import pytest
from paper_agent.state import initial_state


@pytest.mark.asyncio
async def test_full_graph_executes_without_errors():
    state = initial_state("test query")
    mock_search = AsyncMock(return_value={"search_results": [], "errors": [], "trace": []})

    with patch("paper_agent.graph.search_papers", mock_search):
        from paper_agent.graph import build_graph
        app = build_graph()
        result = await app.ainvoke(state, config={"recursion_limit": 50})

    assert result["status"] == "complete"
    assert result["query"] == "test query"
    assert result["ranked_papers"] == []
    assert result["final_review"] != ""


@pytest.mark.asyncio
async def test_graph_short_circuits_compare_for_single_paper():
    state = initial_state("test")
    mock_search = AsyncMock(return_value={"search_results": [], "errors": [], "trace": []})

    with patch("paper_agent.graph.search_papers", mock_search):
        from paper_agent.graph import build_graph
        app = build_graph()
        result = await app.ainvoke(state, config={"recursion_limit": 50})

    assert result["status"] == "complete"
    assert result.get("comparison") is None
