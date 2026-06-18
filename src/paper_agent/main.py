"""CLI entry point for Paper Agent.

Usage:
    paper-agent "What are the latest RAG methods?"
    paper-agent "Compare ColBERT and ColPali" --max-papers 10 --year-from 2022 --year-to 2025
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from paper_agent.config import load_config
from paper_agent.graph import build_graph
from paper_agent.state import initial_state, AgentState


async def run(query: str, *, config_path: str = "config.toml", **overrides) -> AgentState:
    """Execute the paper agent pipeline for a single query.

    Args:
        query: The user's research question.
        config_path: Path to config TOML file.
        **overrides: Optional state field overrides (max_papers, year_range).

    Returns:
        The final AgentState after the graph completes.
    """
    config = load_config(config_path)

    # Build initial state
    state = initial_state(
        query,
        max_papers=overrides.get("max_papers", config.search.max_papers),
        year_range=overrides.get("year_range", config.search.default_year_range),
    )

    # Compile and run the graph
    app = build_graph()
    result = await app.ainvoke(
        state,
        config={"recursion_limit": 50},
    )
    return result


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Paper Agent: AI-powered paper research and survey generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  paper-agent "agentic RAG survey"
  paper-agent "Compare ColBERT and ColPali" --max-papers 10
  paper-agent "Multi-modal document retrieval" --year-from 2023 --year-to 2025
        """,
    )
    parser.add_argument(
        "query",
        nargs="+",
        help="Research question or topic",
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        default=None,
        help="Maximum number of papers to retrieve (default: from config.toml)",
    )
    parser.add_argument(
        "--year-from",
        type=int,
        default=None,
        help="Start year for paper search",
    )
    parser.add_argument(
        "--year-to",
        type=int,
        default=None,
        help="End year for paper search",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.toml",
        help="Path to config TOML file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Override output directory",
    )

    args = parser.parse_args()
    query = " ".join(args.query)

    # Build overrides
    overrides: dict = {}
    if args.max_papers is not None:
        overrides["max_papers"] = args.max_papers
    if args.year_from is not None and args.year_to is not None:
        if args.year_from > args.year_to:
            print("Error: --year-from must be <= --year-to.", file=sys.stderr)
            sys.exit(1)
        overrides["year_range"] = (args.year_from, args.year_to)
    elif args.year_from is not None or args.year_to is not None:
        print("Error: both --year-from and --year-to are required together.", file=sys.stderr)
        sys.exit(1)

    # Run (async)
    print(f"\n🔬 Paper Agent — Researching: '{query}'\n")
    try:
        final_state = asyncio.run(run(query, config_path=args.config, **overrides))
    except KeyboardInterrupt:
        print("\n\n⏹ Interrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}", file=sys.stderr)
        sys.exit(1)

    # Report
    run_id = final_state.get("run_id", "unknown")
    output_dir = Path("outputs/paper-agent") / run_id

    print(f"\n{'─' * 60}")
    print(f"  Run ID:     {run_id}")
    print(f"  Status:     {final_state.get('status', 'unknown')}")
    print(f"  Papers:     {len(final_state.get('ranked_papers', []))}")
    print(f"  Claims:     {len(final_state.get('claims', []))}")
    print(f"  Errors:     {len(final_state.get('errors', []))}")
    print(f"  Output:     {output_dir}")
    print(f"{'─' * 60}\n")

    # Print review if available
    final_review = final_state.get("final_review", "")
    if final_review:
        print(final_review)
    else:
        print("(No review generated — pipeline stubs ran without LLM.)")

    # Print errors
    if final_state.get("errors"):
        print(f"\n⚠️  Errors ({len(final_state['errors'])}):")
        for err in final_state["errors"]:
            print(f"  - {err}")


if __name__ == "__main__":
    main()
