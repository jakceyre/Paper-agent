"""Paper Agent tools — stateless async functions with no LangGraph dependency.

Each tool reads from AgentState and returns a partial state dict.
Tools are independently testable and reusable across agent frameworks.
"""

from paper_agent.tools.search import search_papers, expand_citations
from paper_agent.tools.pdf import download_pdf, parse_pdf
from paper_agent.tools.extract import extract_claims
from paper_agent.tools.compare import compare_papers
from paper_agent.tools.write_review import render_review_markdown

__all__ = [
    "search_papers",
    "expand_citations",
    "download_pdf",
    "parse_pdf",
    "extract_claims",
    "compare_papers",
    "render_review_markdown",
]
