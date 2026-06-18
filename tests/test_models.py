"""Tests for Pydantic data models — validate construction and defaults."""

from paper_agent.models.paper import PaperMetadata, PaperSection, ParsedPaper
from paper_agent.models.claim import Claim, EvidenceItem
from paper_agent.models.comparison import ComparisonTable, ComparisonRow
from paper_agent.models.trace import TraceEvent


def test_paper_metadata_defaults():
    paper = PaperMetadata(paper_id="123", source="arxiv", title="Test", authors=["A"])
    assert paper.year is None
    assert paper.abstract == ""
    assert paper.pdf_url is None


def test_paper_section():
    section = PaperSection(heading="3. Method", level=1, page_start=3, page_end=5, text="Our approach...")
    assert section.heading == "3. Method"
    assert section.page_start == 3


def test_parsed_paper_defaults():
    paper = ParsedPaper(
        paper_id="id",
        metadata=PaperMetadata(paper_id="id", source="arxiv", title="T", authors=["A"]),
    )
    assert paper.sections == []
    assert paper.full_text == ""
    assert paper.page_count == 0


def test_claim_types():
    for ctype in ["contribution", "method", "dataset", "metric", "limitation"]:
        claim = Claim(paper_id="id", claim_type=ctype, claim_text="text", evidence="evidence", section="S1")
        assert claim.claim_type == ctype


def test_comparison_table():
    table = ComparisonTable(
        dimensions=["Method", "Dataset"],
        paper_ids=["p1", "p2"],
        rows=[
            ComparisonRow(dimension="Method", paper_id="p1", value="BERT-based"),
            ComparisonRow(dimension="Method", paper_id="p2", value="T5-based"),
        ],
    )
    assert len(table.rows) == 2
    assert table.paper_ids == ["p1", "p2"]


def test_trace_event():
    event = TraceEvent(timestamp="2025-01-01T00:00:00Z", step="search", event_type="end", data={"count": 5})
    assert event.step == "search"
    assert event.data["count"] == 5
    assert event.error is None
