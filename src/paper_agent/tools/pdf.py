"""PDF tools: download and parse academic papers.

download_pdf: downloads PDFs via httpx with caching and SHA-256 hashing.
parse_pdf: extracts text and detects sections using PyMuPDF (fitz).
"""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import httpx

from paper_agent.config import load_config
from paper_agent.state import AgentState
from paper_agent.models.paper import PaperMetadata, PaperSection, ParsedPaper
from paper_agent.models.trace import TraceEvent

logger = logging.getLogger(__name__)


# ── download_pdf ───────────────────────────────────────


async def download_pdf(state: AgentState) -> dict:
    """Download PDFs for all papers in state.ranked_papers.

    Skips papers without a public pdf_url or already cached.
    Caches PDFs to config.pdf.download_dir.

    Returns partial state dict with:
        - parsed_papers: entries with local_pdf_path and sha256 populated
        - errors: any download failures
        - trace: download events
    """
    now = datetime.now(timezone.utc).isoformat()
    config = load_config()
    ranked: list[PaperMetadata] = state.get("ranked_papers", [])
    download_dir = Path(config.pdf.download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    timeout = config.pdf.download_timeout_sec

    parsed_papers: dict = {}
    errors: list[str] = []
    downloaded = 0
    cached = 0

    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        for paper in ranked:
            if not paper.pdf_url:
                continue

            paper_id = paper.paper_id
            pdf_path = download_dir / f"{paper_id}.pdf"

            try:
                # Check if already cached
                if pdf_path.exists():
                    sha = _sha256_file(pdf_path)
                    cached += 1
                else:
                    # Download
                    response = await client.get(paper.pdf_url)
                    response.raise_for_status()
                    pdf_path.write_bytes(response.content)
                    sha = _sha256_file(pdf_path)
                    downloaded += 1

                parsed_papers[paper_id] = ParsedPaper(
                    paper_id=paper_id,
                    metadata=paper,
                    local_pdf_path=str(pdf_path),
                    sha256=sha,
                )
            except Exception as e:
                msg = f"Download failed for {paper_id}: {e}"
                logger.warning(msg)
                errors.append(msg)

    return {
        "parsed_papers": parsed_papers,
        "errors": errors,
        "trace": [
            TraceEvent(
                timestamp=now,
                step="fetch_pdfs",
                event_type="end",
                data={
                    "total": len(ranked),
                    "downloaded": downloaded,
                    "cached": cached,
                    "failed": len(errors),
                },
            )
        ],
    }


def _sha256_file(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


# ── parse_pdf ──────────────────────────────────────────


async def parse_pdf(state: AgentState) -> dict:
    """Parse downloaded PDFs into structured sections with page numbers.

    Reads state.parsed_papers (populated by download_pdf with local_pdf_path),
    parses each PDF with PyMuPDF (fitz), and returns the updated dict.

    Section detection: uses font-size thresholds and heading-like patterns
    (e.g., "1. Introduction", "Abstract") from get_text("dict") spans.

    Returns partial state dict with:
        - parsed_papers: entries updated with sections, full_text, page_count
        - errors: parse failures
        - trace: parse events
    """
    import fitz  # PyMuPDF — lazy import

    now = datetime.now(timezone.utc).isoformat()
    config = load_config()
    parsed_papers: dict = dict(state.get("parsed_papers", {}))
    max_pages = config.pdf.parse_max_pages
    errors: list[str] = []
    parsed_count = 0

    for paper_id, pp in parsed_papers.items():
        if not pp.local_pdf_path:
            continue

        try:
            doc = fitz.open(pp.local_pdf_path)
            pp.page_count = doc.page_count
            pages_to_read = min(doc.page_count, max_pages)

            full_text_parts: list[str] = []
            sections: list[PaperSection] = []
            current_section: PaperSection | None = None

            for i in range(pages_to_read):
                page = doc.load_page(i)
                page_num = i + 1  # 1-indexed
                page_text = page.get_text("text")
                full_text_parts.append(page_text)

                # Detect headings from structured dict
                headings = _detect_headings(page, page_num)
                if headings:
                    # Close previous section
                    if current_section is not None:
                        current_section.text = current_section.text.rstrip()
                        current_section.page_end = max(page_num - 1, current_section.page_start)
                        sections.append(current_section)
                    # Start new section
                    current_section = headings[0]
                elif current_section is not None:
                    # Accumulate text into current section
                    if current_section.text:
                        current_section.text += "\n"
                    current_section.text += page_text
                    current_section.page_end = page_num

            # Close last section
            if current_section is not None:
                current_section.text = current_section.text.rstrip()
                sections.append(current_section)

            # If no sections detected, create one per page
            if not sections:
                for i in range(pages_to_read):
                    sections.append(
                        PaperSection(
                            heading=f"Page {i + 1}",
                            level=1,
                            page_start=i + 1,
                            page_end=i + 1,
                            text=full_text_parts[i],
                        )
                    )

            pp.full_text = "\n".join(full_text_parts)
            pp.sections = sections
            doc.close()
            parsed_count += 1

        except Exception as e:
            msg = f"Parse failed for {paper_id}: {e}"
            logger.error(msg)
            errors.append(msg)

    return {
        "parsed_papers": parsed_papers,
        "errors": errors,
        "trace": [
            TraceEvent(
                timestamp=now,
                step="parse_pdfs",
                event_type="end",
                data={
                    "total": len(parsed_papers),
                    "parsed": parsed_count,
                    "failed": len(errors),
                },
            )
        ],
    }


def _detect_headings(page, page_num: int) -> list[PaperSection]:
    """Detect section headings on a page using font-size and pattern heuristics.

    Returns a list of PaperSection objects — one per detected heading.
    The caller handles section continuity across pages.
    """
    dict_data = page.get_text("dict")
    blocks = dict_data.get("blocks", [])

    # Collect all spans to compute median font size
    all_sizes: list[float] = []

    for block in blocks:
        if block.get("type") != 0:  # text blocks only
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "").strip()
                if text:
                    all_sizes.append(span.get("size", 11))

    if not all_sizes:
        return []

    sorted_sizes = sorted(all_sizes)
    n = len(sorted_sizes)
    median_size = sorted_sizes[n // 2] if n > 0 else 11.0

    detected: list[PaperSection] = []

    for block in blocks:
        if block.get("type") != 0:
            continue

        block_text = ""
        block_size = 0.0
        block_flags = 0

        for line in block.get("lines", []):
            line_text = ""
            for span in line.get("spans", []):
                line_text += span.get("text", "")
                if not block_size:
                    block_size = span.get("size", 11)
                    block_flags = span.get("flags", 0)
            block_text += line_text.rstrip() + " "

        block_text = block_text.strip()
        if not block_text or len(block_text) < 3:
            continue

        # Heuristics: larger font OR bold flag + heading-like pattern
        is_bold = bool(block_flags & 8)  # bit 3 = bold
        is_larger = block_size >= median_size * 1.15 if median_size > 0 else block_size >= 13

        heading_pattern = bool(
            re.match(r"^(\d+\.?)+(\s+[A-Z])", block_text)
            or block_text.lower().startswith(("abstract", "introduction", "conclusion",
                                               "references", "acknowledgment"))
        )

        if (is_larger or is_bold) and heading_pattern:
            level = 2 if re.match(r"^\d+\.\d+", block_text) else 1
            detected.append(
                PaperSection(
                    heading=block_text[:120],
                    level=level,
                    page_start=page_num,
                    page_end=page_num,
                    text="",
                )
            )

    return detected
