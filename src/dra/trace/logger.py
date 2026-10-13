"""JSONL trace：每次运行落一条，喂给 evals——观测和评估共用一套底座。"""
from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RunTrace:
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    question: str = ""
    started_at: float = field(default_factory=time.time)
    finished_at: float = 0.0
    subquestions: list[str] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    token_usage: dict[str, int] = field(default_factory=dict)  # node -> tokens
    llm_calls: int = 0
    verify_supported: int = 0
    verify_partial: int = 0
    verify_unsupported: int = 0
    budget_exhausted: bool = False
    error: str = ""

    @property
    def total_tokens(self) -> int:
        return sum(self.token_usage.values())

    @property
    def latency_s(self) -> float:
        return (self.finished_at or time.time()) - self.started_at


class TraceLogger:
    """线程安全的 JSONL 追加写。同时是 SSE 事件源。"""

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def finish(self, trace: RunTrace) -> None:
        trace.finished_at = time.time()
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(trace), ensure_ascii=False) + "\n")

    def read_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
