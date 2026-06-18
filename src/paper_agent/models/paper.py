"""Paper data models: metadata, sections, and full parsed papers."""

from __future__ import annotations

from pydantic import BaseModel


class PaperMetadata(BaseModel):
    """Lightweight paper metadata from search APIs, before PDF download."""

    paper_id: str
    source: str
    title: str
    authors: list[str]
    year: int | None = None
    abstract: str = ""
    url: str = ""
    pdf_url: str | None = None
    citation_count: int | None = None
    venue: str | None = None


class PaperSection(BaseModel):
    """A parsed section within a paper, with page range and text."""

    heading: str
    level: int = 1
    page_start: int
    page_end: int
    text: str


class ParsedPaper(BaseModel):
    """Full parsed paper content, keyed by paper_id in state."""

    paper_id: str
    metadata: PaperMetadata
    sections: list[PaperSection] = []
    full_text: str = ""
    page_count: int = 0
    local_pdf_path: str | None = None
    sha256: str | None = None
