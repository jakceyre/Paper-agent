"""Evidence store: append-only JSONL for auditable evidence trail."""

from __future__ import annotations

import json
from pathlib import Path

from paper_agent.models.claim import EvidenceItem


class EvidenceStore:
    """Append-only JSONL store for evidence items."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, item: EvidenceItem) -> None:
        with open(self.path, "a") as f:
            f.write(item.model_dump_json() + "\n")

    def append_batch(self, items: list[EvidenceItem]) -> None:
        with open(self.path, "a") as f:
            for item in items:
                f.write(item.model_dump_json() + "\n")

    def read_all(self) -> list[EvidenceItem]:
        if not self.path.exists():
            return []
        items = []
        with open(self.path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        items.append(EvidenceItem(**json.loads(line)))
                    except Exception:
                        continue
        return items

    def count(self) -> int:
        if not self.path.exists():
            return 0
        with open(self.path, "r") as f:
            return sum(1 for _ in f)
