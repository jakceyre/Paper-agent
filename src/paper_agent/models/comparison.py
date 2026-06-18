"""Comparison models for cross-paper analysis."""

from __future__ import annotations

from pydantic import BaseModel


class ComparisonRow(BaseModel):
    """One row in a comparison table: a single dimension for a single paper."""

    dimension: str
    paper_id: str
    value: str


class ComparisonTable(BaseModel):
    """Structured comparison across multiple papers."""

    dimensions: list[str]
    paper_ids: list[str]
    rows: list[ComparisonRow] = []
    caveats: str | None = None
