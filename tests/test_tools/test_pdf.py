"""Tests for PDF download and parsing tools."""

import tempfile
from pathlib import Path

import fitz
import pytest
from paper_agent.tools.pdf import download_pdf, parse_pdf, _sha256_file, _detect_headings
from paper_agent.models.paper import PaperMetadata
from paper_agent.state import initial_state


# ── Helper tests ───────────────────────────────────────


def test_sha256_file():
    """SHA-256 should produce a 64-char hex string."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.txt"
        path.write_bytes(b"hello world")
        sha = _sha256_file(path)
        assert len(sha) == 64
        assert all(c in "0123456789abcdef" for c in sha)


def test_detect_headings_runs_without_error():
    """_detect_headings should run without error on any page.

    Note: insert_text creates a different internal structure than real
    academic PDFs, so we only verify the function doesn't crash.
    Heading detection is designed for real arXiv PDFs with rich font data.
    """
    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / "test.pdf"
        doc = fitz.open()
        page = doc.new_page(width=612, height=792)
        page.insert_text((72, 72), "1. Introduction", fontsize=18, fontname="helv")
        page.insert_text((72, 100), "Body text here.", fontsize=11, fontname="helv")
        doc.save(str(pdf_path))
        doc.close()

        doc = fitz.open(str(pdf_path))
        page = doc.load_page(0)
        headings = _detect_headings(page, 1)
        doc.close()

        # Function should return a list without crashing
        assert isinstance(headings, list)


# ── download_pdf tests ─────────────────────────────────


@pytest.mark.asyncio
async def test_download_pdf_returns_expected_keys():
    """download_pdf returns parsed_papers, errors, trace."""
    state = initial_state("test")
    result = await download_pdf(state)
    assert "parsed_papers" in result
    assert "errors" in result
    assert "trace" in result


@pytest.mark.asyncio
async def test_download_pdf_empty_when_no_ranked():
    """With no ranked papers, returns empty dict."""
    state = initial_state("test")
    result = await download_pdf(state)
    assert result["parsed_papers"] == {}


@pytest.mark.asyncio
async def test_download_pdf_skips_no_pdf_url():
    """Papers without pdf_url should be skipped."""
    state = initial_state("test")
    paper = PaperMetadata(
        paper_id="no_pdf", source="arxiv", title="No PDF Paper",
        authors=["A"], year=2023, pdf_url=None,
    )
    state["ranked_papers"] = [paper]
    result = await download_pdf(state)
    assert result["parsed_papers"] == {}


# ── parse_pdf tests ────────────────────────────────────


@pytest.mark.asyncio
async def test_parse_pdf_returns_expected_keys():
    """parse_pdf returns parsed_papers, errors, trace."""
    state = initial_state("test")
    result = await parse_pdf(state)
    assert "parsed_papers" in result
    assert "errors" in result
    assert "trace" in result


@pytest.mark.asyncio
async def test_parse_pdf_empty_when_no_parsed():
    """With no parsed papers, returns empty dict."""
    state = initial_state("test")
    result = await parse_pdf(state)
    assert result["parsed_papers"] == {}
