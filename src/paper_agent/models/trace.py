"""Trace event model for observability and eval."""

from __future__ import annotations

from pydantic import BaseModel


class TraceEvent(BaseModel):
    """A single trace event recording a step in the pipeline."""

    timestamp: str
    step: str
    event_type: str
    data: dict = {}
    error: str | None = None
