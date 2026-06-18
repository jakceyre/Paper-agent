"""Pydantic data models for Paper Agent.

These are pure data containers with no business logic.
They define the boundaries between tools, agents, and storage.
"""

from paper_agent.models.paper import PaperMetadata, PaperSection, ParsedPaper
from paper_agent.models.claim import Claim, EvidenceItem
from paper_agent.models.comparison import ComparisonRow, ComparisonTable
from paper_agent.models.trace import TraceEvent

__all__ = [
    "PaperMetadata",
    "PaperSection",
    "ParsedPaper",
    "Claim",
    "EvidenceItem",
    "ComparisonRow",
    "ComparisonTable",
    "TraceEvent",
]
