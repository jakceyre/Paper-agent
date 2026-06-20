"""Eval runner: executes eval cases through the Paper Agent pipeline.

Usage:
    python -m eval.runner                    # Run all cases
    python -m eval.runner --type topic_search  # Run only topic_search cases
    python -m eval.runner --case topic_01    # Run a single case
    python -m eval.runner --max 5            # Run first 5 cases only
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from eval.cases import CASES, get_cases

# Add project root to path (for paper_agent import)
sys.path.insert(0, str(Path(__file__).parent.parent))

from paper_agent.main import run  # noqa: E402


async def run_case(case, output_dir: Path) -> dict:
    """Run a single eval case and return summary stats."""
    case_dir = output_dir / case.case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    start = time.monotonic()
    try:
        state = await run(
            case.query,
            config_path="config.toml",
            max_papers=case.max_papers,
            year_range=case.year_range,
        )
        elapsed = time.monotonic() - start

        # Move outputs to case dir
        run_id = state.get("run_id", "unknown")
        default_dir = Path("outputs/paper-agent") / run_id
        if default_dir.exists():
            for f in default_dir.iterdir():
                f.rename(case_dir / f.name)
            default_dir.rmdir()

        return {
            "case_id": case.case_id,
            "case_type": case.case_type,
            "status": state.get("status", "unknown"),
            "papers_found": len(state.get("ranked_papers", [])),
            "claims_extracted": len(state.get("claims", [])),
            "errors_count": len(state.get("errors", [])),
            "latency_sec": round(elapsed, 2),
            "run_id": run_id,
            "query": case.query,
        }
    except Exception as e:
        elapsed = time.monotonic() - start
        return {
            "case_id": case.case_id,
            "case_type": case.case_type,
            "status": "error",
            "error": str(e),
            "latency_sec": round(elapsed, 2),
            "query": case.query,
        }


async def main(args):
    """Run eval cases."""
    if args.case:
        case_obj = next((c for c in CASES if c.case_id == args.case), None)
        if not case_obj:
            print(f"Unknown case: {args.case}")
            return
        cases = [case_obj]
    elif args.type:
        cases = get_cases([args.type])
    else:
        cases = list(CASES)

    if args.max_cases:
        cases = cases[: args.max_cases]

    output_dir = Path("outputs/eval") / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running {len(cases)} eval cases...")
    print(f"Output dir: {output_dir}\n")

    results = []
    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {case.case_id}: {case.query[:60]}...")
        result = await run_case(case, output_dir)
        results.append(result)
        status_icon = "✅" if result.get("status") == "complete" else "❌"
        papers = result.get("papers_found", 0)
        claims = result.get("claims_extracted", 0)
        latency = result.get("latency_sec", 0)
        print(f"    {status_icon} papers={papers} claims={claims} {latency}s\n")

    # Write summary
    summary_path = output_dir / "eval_summary.json"
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_cases": len(results),
        "completed": sum(1 for r in results if r.get("status") == "complete"),
        "errored": sum(1 for r in results if r.get("status") == "error"),
        "avg_latency": round(
            sum(r.get("latency_sec", 0) for r in results) / len(results), 2
        ) if results else 0,
        "total_papers": sum(r.get("papers_found", 0) for r in results),
        "total_claims": sum(r.get("claims_extracted", 0) for r in results),
        "results": results,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Summary: {summary_path}")
    print(f"  Completed: {summary['completed']}/{summary['total_cases']}")
    print(f"  Avg latency: {summary['avg_latency']}s")
    print(f"  Total papers: {summary['total_papers']}")
    print(f"  Total claims: {summary['total_claims']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Paper Agent Eval Runner")
    parser.add_argument("--case", type=str, help="Run a single case by ID")
    parser.add_argument("--type", type=str, help="Run cases of a specific type")
    parser.add_argument("--max-cases", type=int, default=None, help="Max cases to run")
    args = parser.parse_args()
    asyncio.run(main(args))
