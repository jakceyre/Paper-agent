"""Tests for PDF download and parsing tools."""

import pytest
from paper_agent.tools.pdf import download_pdf, parse_pdf
from paper_agent.state import initial_state


@pytest.mark.asyncio
async def test_download_pdf_returns_expected_keys():
    state = initial_state("test")
    result = await download_pdf(state)
    assert "parsed_papers" in result
    assert "errors" in result
    assert "trace" in result


@pytest.mark.asyncio
async def test_download_pdf_empty_when_no_ranked():
    state = initial_state("test")
    result = await download_pdf(state)
    assert result["parsed_papers"] == {}


@pytest.mark.asyncio
async def test_parse_pdf_returns_expected_keys():
    state = initial_state("test")
    result = await parse_pdf(state)
    assert "parsed_papers" in result
    assert "errors" in result
    assert "trace" in result
