"""SQLite store for paper metadata and run history."""

from __future__ import annotations

import sqlite3
from pathlib import Path


class SQLiteStore:
    """Manages SQLite database for paper metadata and run history."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS papers (
        paper_id TEXT PRIMARY KEY,
        source TEXT NOT NULL,
        title TEXT NOT NULL,
        authors TEXT,
        year INTEGER,
        abstract TEXT,
        url TEXT,
        pdf_url TEXT,
        citation_count INTEGER,
        venue TEXT,
        created_at TEXT DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS runs (
        run_id TEXT PRIMARY KEY,
        query TEXT NOT NULL,
        status TEXT DEFAULT 'running',
        papers_found INTEGER DEFAULT 0,
        claims_extracted INTEGER DEFAULT 0,
        errors_count INTEGER DEFAULT 0,
        created_at TEXT DEFAULT (datetime('now')),
        completed_at TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_papers_year ON papers(year);
    CREATE INDEX IF NOT EXISTS idx_papers_source ON papers(source);
    CREATE INDEX IF NOT EXISTS idx_runs_created ON runs(created_at);
    """

    def __init__(self, db_path: str | Path = "outputs/paper-agent/store.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
            self._conn.executescript(self.SCHEMA)
            self._conn.commit()
        return self._conn

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def upsert_paper(self, paper: dict) -> None:
        import json

        self.conn.execute(
            """INSERT OR REPLACE INTO papers
               (paper_id, source, title, authors, year, abstract, url, pdf_url, citation_count, venue)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                paper["paper_id"],
                paper.get("source", ""),
                paper["title"],
                json.dumps(paper.get("authors", [])),
                paper.get("year"),
                paper.get("abstract", ""),
                paper.get("url", ""),
                paper.get("pdf_url"),
                paper.get("citation_count"),
                paper.get("venue"),
            ),
        )
        self.conn.commit()

    def upsert_paper_model(self, paper) -> None:
        self.upsert_paper(paper.model_dump())

    def get_paper(self, paper_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM papers WHERE paper_id = ?", (paper_id,)
        ).fetchone()
        return dict(row) if row else None

    def list_papers(self, limit: int = 100) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM papers ORDER BY year DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def start_run(self, run_id: str, query: str) -> None:
        self.conn.execute(
            "INSERT INTO runs (run_id, query) VALUES (?, ?)",
            (run_id, query),
        )
        self.conn.commit()

    def update_run(self, run_id: str, **kwargs) -> None:
        if not kwargs:
            return
        sets = ", ".join(f"{k} = ?" for k in kwargs)
        values = list(kwargs.values()) + [run_id]
        self.conn.execute(f"UPDATE runs SET {sets} WHERE run_id = ?", values)
        self.conn.commit()

    def complete_run(self, run_id: str, papers_found: int, claims_extracted: int, errors_count: int) -> None:
        self.update_run(
            run_id,
            status="complete",
            papers_found=papers_found,
            claims_extracted=claims_extracted,
            errors_count=errors_count,
            completed_at="datetime('now')",
        )
