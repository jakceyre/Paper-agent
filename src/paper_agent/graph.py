"""LangGraph StateGraph assembly — the core pipeline.

Graph structure (MVP, deterministic edges):

    START -> plan -> search -> rank -> fetch_pdfs -> parse_pdfs
        -> extract -> [conditional: >=2 papers?]
            YES -> compare -> synthesize
            NO  -> synthesize
        -> review_critique -> write_outputs -> END

All nodes are async functions that take AgentState and return a partial dict.
"""

from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from paper_agent.state import AgentState
from paper_agent.agents.planner import plan
from paper_agent.agents.ranker import rank
from paper_agent.agents.synthesizer import synthesize
from paper_agent.agents.reviewer import review
from paper_agent.tools.search import search_papers
from paper_agent.tools.pdf import download_pdf, parse_pdf
from paper_agent.tools.extract import extract_claims
from paper_agent.tools.compare import compare_papers
from paper_agent.store.output_writer import write_outputs


def build_graph() -> StateGraph:
    """Construct and return the compiled LangGraph StateGraph.

    Returns:
        A compiled StateGraph ready for ainvoke().
    """
    builder = StateGraph(AgentState)

    # ── Add nodes ──────────────────────────────────────
    builder.add_node("plan", plan)
    builder.add_node("search", search_papers)
    builder.add_node("rank", rank)
    builder.add_node("fetch_pdfs", download_pdf)
    builder.add_node("parse_pdfs", parse_pdf)
    builder.add_node("extract", extract_claims)
    builder.add_node("compare", compare_papers)
    builder.add_node("synthesize", synthesize)
    builder.add_node("review_critique", review)
    builder.add_node("write_outputs", write_outputs)

    # ── Add edges ──────────────────────────────────────
    builder.add_edge(START, "plan")
    builder.add_edge("plan", "search")
    builder.add_edge("search", "rank")
    builder.add_edge("rank", "fetch_pdfs")
    builder.add_edge("fetch_pdfs", "parse_pdfs")
    builder.add_edge("parse_pdfs", "extract")

    # Conditional branch: only compare when >= 2 papers have claims
    builder.add_conditional_edges(
        "extract",
        _should_compare,
        {
            "compare": "compare",
            "synthesize": "synthesize",
        },
    )
    builder.add_edge("compare", "synthesize")
    builder.add_edge("synthesize", "review_critique")
    builder.add_edge("review_critique", "write_outputs")
    builder.add_edge("write_outputs", END)

    return builder.compile()


def _should_compare(state: AgentState) -> str:
    """Conditional routing: only run compare when >= 2 papers have claims."""
    claims = state.get("claims", [])
    paper_ids = set(c.paper_id for c in claims)
    return "compare" if len(paper_ids) >= 2 else "synthesize"
