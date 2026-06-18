"""Tests for evidence store (JSONL append-only)."""

import tempfile
from pathlib import Path

from paper_agent.store.evidence_store import EvidenceStore
from paper_agent.models.claim import EvidenceItem


def test_append_and_read():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "evidence.jsonl"
        store = EvidenceStore(path)

        item = EvidenceItem(
            evidence_id="ev_001", paper_id="2301.12345", claim_text="A novel RAG architecture",
            quote="We propose...", section="3. Method", page=3,
            url="https://arxiv.org/abs/2301.12345", timestamp="2025-01-01T00:00:00Z",
        )

        store.append(item)
        assert store.count() == 1

        items = store.read_all()
        assert len(items) == 1
        assert items[0].evidence_id == "ev_001"
        assert items[0].paper_id == "2301.12345"


def test_append_batch():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "evidence.jsonl"
        store = EvidenceStore(path)

        items = [
            EvidenceItem(
                evidence_id=f"ev_{i:03d}", paper_id="2301.12345", claim_text=f"Claim {i}",
                timestamp="2025-01-01T00:00:00Z",
            )
            for i in range(10)
        ]

        store.append_batch(items)
        assert store.count() == 10


def test_read_empty_store():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "nonexistent.jsonl"
        store = EvidenceStore(path)
        assert store.read_all() == []
        assert store.count() == 0
