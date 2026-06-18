"""Output writer: writes all final artifacts to the run directory.

Produces:
    outputs/paper-agent/<run-id>/
    ├── review.md
    ├── papers.bib
    ├── evidence.jsonl
    ├── comparison.csv
    ├── trace.jsonl
    └── eval_summary.md
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from paper_agent.state import AgentState
from paper_agent.models.trace import TraceEvent
from paper_agent.store.evidence_store import EvidenceStore


async def write_outputs(state: AgentState) -> dict:
    """Write all output artifacts for the completed run."""
    now = datetime.now(timezone.utc).isoformat()
    run_id = state["run_id"]
    base_dir = Path("outputs/paper-agent") / run_id
    base_dir.mkdir(parents=True, exist_ok=True)

    errors: list[str] = []

    # 1. Write review.md
    try:
        review_path = base_dir / "review.md"
        review_path.write_text(state.get("final_review", ""), encoding="utf-8")
    except Exception as e:
        errors.append(f"Failed to write review.md: {e}")

    # 2. Write evidence.jsonl
    try:
        evidence_path = base_dir / "evidence.jsonl"
        store = EvidenceStore(evidence_path)
        store.append_batch(state.get("evidence", []))
    except Exception as e:
        errors.append(f"Failed to write evidence.jsonl: {e}")

    # 3. Write comparison.csv
    try:
        comparison = state.get("comparison")
        if comparison and comparison.rows:
            csv_path = base_dir / "comparison.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Dimension", "Paper ID", "Value"])
                for row in comparison.rows:
                    writer.writerow([row.dimension, row.paper_id, row.value])
    except Exception as e:
        errors.append(f"Failed to write comparison.csv: {e}")

    # 4. Write papers.bib
    try:
        bib_path = base_dir / "papers.bib"
        bib_entries = state.get("bibtex_entries", [])
        if bib_entries:
            bib_path.write_text("\n\n".join(bib_entries), encoding="utf-8")
    except Exception as e:
        errors.append(f"Failed to write papers.bib: {e}")

    # 5. Write trace.jsonl
    try:
        trace_path = base_dir / "trace.jsonl"
        trace_events = state.get("trace", [])
        with open(trace_path, "w", encoding="utf-8") as f:
            for event in trace_events:
                if isinstance(event, TraceEvent):
                    f.write(event.model_dump_json() + "\n")
                elif isinstance(event, dict):
                    f.write(json.dumps(event) + "\n")
    except Exception as e:
        errors.append(f"Failed to write trace.jsonl: {e}")

    # 6. Write eval_summary.md (stub)
    try:
        eval_path = base_dir / "eval_summary.md"
        eval_text = _render_eval_summary(state, errors)
        eval_path.write_text(eval_text, encoding="utf-8")
    except Exception as e:
        errors.append(f"Failed to write eval_summary.md: {e}")

    return {
        "errors": errors,
        "trace": [
            TraceEvent(
                timestamp=now,
                step="write_outputs",
                event_type="end",
                data={
                    "output_dir": str(base_dir),
                    "files_written": 6 - len(errors),
                },
            )
        ],
    }


def _render_eval_summary(state: AgentState, write_errors: list[str]) -> str:
    """Render a brief evaluation summary in Markdown."""
    lines = [
        f"# Eval Summary: {state.get('query', 'Unknown')}",
        "",
        f"**Run ID**: `{state.get('run_id', 'N/A')}`",
        f"**Status**: {state.get('status', 'unknown')}",
        "",
        "## Stats",
        "",
        f"- Papers found: {len(state.get('ranked_papers', []))}",
        f"- Claims extracted: {len(state.get('claims', []))}",
        f"- Errors: {len(state.get('errors', []))}",
        f"- Write errors: {len(write_errors)}",
        "",
    ]

    errors = state.get("errors", []) + write_errors
    if errors:
        lines.append("## Errors")
        lines.append("")
        for e in errors:
            lines.append(f"- {e}")
        lines.append("")

    return "\n".join(lines)
