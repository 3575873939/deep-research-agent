"""reranker：可选 cross-encoder（bge-reranker），无依赖时退化为 RRF 顺序。

退化策略是刻意的：BM25-only 模式必须全链路可跑（CI、无 GPU、面试现场断网）。
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .ingest import Chunk


class Reranker:
    """bge-reranker cross-encoder 包装；模型不可用时 fallback()。"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base", device: str = "cpu") -> None:
        self._model = None
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(model_name, device=device)
        except Exception:  # noqa: BLE001 - 依赖/模型缺失都走 fallback，这是设计
            self._model = None

    @property
    def available(self) -> bool:
        return self._model is not None

    def rerank(
        self, query: str, candidates: list[tuple["Chunk", float]], top_k: int = 5
    ) -> list[tuple["Chunk", float]]:
        if not candidates:
            return []
        if self._model is None:
            return self.fallback(candidates, top_k)
        pairs = [(query, c.text) for c, _ in candidates]
        scores = self._model.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: float(x[1]), reverse=True)
        return [(c, float(s)) for (c, _), s in ranked[:top_k]]

    @staticmethod
    def fallback(candidates: list[tuple["Chunk", float]], top_k: int) -> list[tuple["Chunk", float]]:
        return candidates[:top_k]
