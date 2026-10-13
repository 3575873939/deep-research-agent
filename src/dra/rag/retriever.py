"""混合检索：BM25 + 向量各取 top10 → RRF 融合 top20 → (可选) reranker 取 top5。

面试问答准备：
- 纯向量在专有名词/编号上召回差，BM25 补精确匹配
- RRF 免调权（不需要调 alpha 混合系数），对分数尺度不敏感
- 向量与 reranker 均为可选依赖：BM25-only 模式全链路可跑（CI/无 GPU 环境）
"""
from __future__ import annotations

import math
import re
from typing import Protocol

from rank_bm25 import BM25Plus

from .ingest import Chunk


def tokenize(text: str) -> list[str]:
    """中英混合分词：英文按词，中文按 bigram（免 jieba 依赖，召回足够）。"""
    text = text.lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    cjk = re.findall(r"[一-鿿]", text)
    tokens += cjk + [a + b for a, b in zip(cjk, cjk[1:])]
    return tokens


class VectorIndex(Protocol):
    def search(self, query: str, top_k: int) -> list[tuple[int, float]]: ...


def rrf_fuse(rankings: list[list[int]], *, k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion：score = Σ 1/(k + rank)。"""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


class HybridRetriever:
    """进程内检索器。研究 Agent 每 run 的语料是中小规模（几十页），进程内索引足够。"""

    def __init__(self, chunks: list[Chunk], vector_index: VectorIndex | None = None) -> None:
        self.chunks = chunks
        self.vector_index = vector_index
        self._bm25: BM25Plus | None = None
        self._token_sets: list[set[str]] = []
        self._rebuild()

    def _rebuild(self) -> None:
        """重建索引。同时缓存词项集合——BM25Plus 的 delta 项会给零命中文档非零分，
        检索时需按词项重叠过滤，避免无关块被选入证据。"""
        tokens = [tokenize(c.text) for c in self.chunks]
        self._bm25 = BM25Plus(tokens) if tokens else None
        self._token_sets = [set(t) for t in tokens]

    def add(self, chunks: list[Chunk]) -> None:
        self.chunks.extend(chunks)
        self._rebuild()

    def retrieve(self, query: str, *, top_k: int = 20) -> list[tuple[Chunk, float]]:
        if not self.chunks or self._bm25 is None:
            return []
        rankings: list[list[int]] = []

        # BM25 top10（按词项重叠过滤零命中）
        q_tokens = set(tokenize(query))
        bm25_scores = self._bm25.get_scores(tokenize(query))
        bm25_top = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:10]
        rankings.append([i for i in bm25_top if self._token_sets[i] & q_tokens])

        # 向量 top10（可选）
        if self.vector_index is not None:
            vec_top = self.vector_index.search(query, 10)
            rankings.append([i for i, _ in vec_top])

        fused = rrf_fuse([r for r in rankings if r])
        return [(self.chunks[i], score) for i, score in fused[:top_k]]
