"""引用库：SQLite 持久化"原文快照"。[cite:N] 的 N 对应这里的行 id，机器可查。

引用可验证的地基：每条引用带 url + 原文快照 + 内容哈希（溯源用）。
"""
from __future__ import annotations

import hashlib
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS citations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    url TEXT NOT NULL,
    title TEXT DEFAULT '',
    snapshot TEXT NOT NULL,          -- 原文快照（进入 context 的原文）
    content_hash TEXT NOT NULL,      -- sha256(snapshot)，溯源校验
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_citations_run ON citations(run_id);
"""


@dataclass
class Citation:
    id: int
    url: str
    title: str
    snapshot: str
    content_hash: str


class CitationStore:
    def __init__(self, db_path: str) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def add(self, run_id: str, url: str, title: str, snapshot: str) -> int:
        h = hashlib.sha256(snapshot.encode("utf-8")).hexdigest()
        cur = self._conn.execute(
            "INSERT INTO citations(run_id, url, title, snapshot, content_hash, created_at)"
            " VALUES (?,?,?,?,?,?)",
            (run_id, url, title, snapshot, h, time.time()),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def get(self, citation_id: int) -> Citation | None:
        row = self._conn.execute(
            "SELECT id, url, title, snapshot, content_hash FROM citations WHERE id=?",
            (citation_id,),
        ).fetchone()
        if not row:
            return None
        return Citation(id=row[0], url=row[1], title=row[2], snapshot=row[3], content_hash=row[4])

    def list_for_run(self, run_id: str) -> list[Citation]:
        rows = self._conn.execute(
            "SELECT id, url, title, snapshot, content_hash FROM citations WHERE run_id=? ORDER BY id",
            (run_id,),
        ).fetchall()
        return [Citation(*r) for r in rows]

    def close(self) -> None:
        self._conn.close()
