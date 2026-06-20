"""Eval analyzer: reads trace.jsonl files and computes metrics.

Usage:
    python -m eval.analyze outputs/eval/20260101_120000
    python -m eval.analyze outputs/eval/20260101_120000 --metrics all
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def analyze(eval_dir: str | Path) -> dict:
    """Analyze all case outputs in an eval directory.

    Returns a dict with:
        - per_case_stats: aggregated numbers per case
        - overall: overall metrics across all cases
        - by_type: metrics grouped by case_type
    """
    root = Path(eval_dir)
    if not root.exists():
        return {"error": f"Directory not found: {eval_dir}"}

    # Find all trace.jsonl files
    case_stats: dict[str, dict] = {}
    all_steps: list[dict] = []

    for case_dir in sorted(root.iterdir()):
        if not case_dir.is_dir():
            continue
        trace_file = case_dir / "trace.jsonl"
        if not trace_file.exists():
            continue

        case_id = case_dir.name
        steps = _parse_trace(trace_file)
        all_steps.extend(steps)

        # Per-case stats
        errors = [s for s in steps if s.get("event_type") == "error" or s.get("error")]
        case_stats[case_id] = {
            "total_steps": len(steps),
            "error_steps": len(errors),
            "nodes_visited": list(dict.fromkeys(s["step"] for s in steps)),
        }

    # Overall metrics
    error_steps = [s for s in all_steps if s.get("event_type") == "error" or s.get("error")]
    node_counts = defaultdict(int)
    for s in all_steps:
        node_counts[s.get("step", "unknown")] += 1

    return {
        "total_cases": len(case_stats),
        "total_trace_events": len(all_steps),
        "error_rate": round(len(error_steps) / len(all_steps), 4) if all_steps else 0,
        "nodes_visited": dict(node_counts),
        "per_case": case_stats,
    }


def _parse_trace(path: Path) -> list[dict]:
    """Parse a trace.jsonl file into a list of event dicts."""
    events = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return events


def compute_metrics(results_dir: Path) -> dict:
    """Compute the 5 README metrics from eval outputs.

    Metrics:
        paper_relevance@k — requires human judgment (placeholder)
        citation_precision — claim with non-empty evidence / total claims
        coverage_score — unique claim_types / 5 (method, dataset, metric, contribution, limitation)
        hallucination_rate — claims with empty evidence / total claims
        avg_cost / avg_latency / avg_steps — from trace events
    """
    metrics: dict = {
        "paper_relevance@5": None,  # requires human annotation
        "citation_precision": 0.0,
        "coverage_score": 0.0,
        "hallucination_rate": 0.0,
        "avg_steps": 0,
        "avg_latency_sec": 0,
    }

    total_claims = 0
    claims_with_evidence = 0
    claim_types_seen: set[str] = set()
    total_steps = 0
    total_latency = 0.0
    case_count = 0

    for case_dir in sorted(results_dir.iterdir()):
        if not case_dir.is_dir():
            continue
        case_count += 1

        # Parse evidence.jsonl for claims stats
        evidence_file = case_dir / "evidence.jsonl"
        if evidence_file.exists():
            with open(evidence_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ev = json.loads(line)
                        total_claims += 1
                        if ev.get("quote"):
                            claims_with_evidence += 1
                    except json.JSONDecodeError:
                        continue

        # Parse trace.jsonl for step/latency stats
        trace_file = case_dir / "trace.jsonl"
        if trace_file.exists():
            events = _parse_trace(trace_file)
            total_steps += len(events)
            if events:
                first = events[0].get("timestamp", "")
                last = events[-1].get("timestamp", "")
                if first and last:
                    try:
                        from datetime import datetime
                        t1 = datetime.fromisoformat(first)
                        t2 = datetime.fromisoformat(last)
                        total_latency += (t2 - t1).total_seconds()
                    except (ValueError, TypeError):
                        pass

    if total_claims > 0:
        metrics["citation_precision"] = round(claims_with_evidence / total_claims, 4)
        metrics["hallucination_rate"] = round((total_claims - claims_with_evidence) / total_claims, 4)

    expected_types = {"method", "dataset", "metric", "contribution", "limitation"}
    metrics["coverage_score"] = round(len(claim_types_seen) / len(expected_types), 4)

    if case_count > 0:
        metrics["avg_steps"] = round(total_steps / case_count, 1)
        metrics["avg_latency_sec"] = round(total_latency / case_count, 2)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Paper Agent Eval Analyzer")
    parser.add_argument("eval_dir", help="Path to eval output directory")
    parser.add_argument("--metrics", action="store_true", help="Compute README metrics")
    args = parser.parse_args()

    result = analyze(args.eval_dir)
    print(json.dumps(result, indent=2, default=str))

    if args.metrics:
        metrics = compute_metrics(Path(args.eval_dir))
        print("\n📊 README Metrics:")
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
