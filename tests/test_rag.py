"""RRF 融合 + 切分 + 混合检索（BM25-only 模式）+ researcher 证据选块集成。"""
from dra.config import Settings
from dra.graph.researcher import _build_evidence, _fallback_evidence
from dra.rag.ingest import chunk_text
from dra.rag.retriever import HybridRetriever, rrf_fuse, tokenize


def test_rrf_prefers_multi_list_hits():
    # doc1 在两个列表都靠前，doc2 只在一个列表靠前 → doc1 胜出
    fused = rrf_fuse([[0, 1, 2], [0, 3, 4]])
    assert fused[0][0] == 0


def test_rrf_empty():
    assert rrf_fuse([]) == []


def test_tokenize_mixed():
    toks = tokenize("BM25 检索 retrieval")
    assert "bm25" in toks and "retrieval" in toks
    assert any(len(t) == 2 for t in toks)  # 中文 bigram


def test_chunk_overlap_and_hash():
    text = "。".join(f"第{i}句" * 100 for i in range(30))
    chunks = chunk_text(text, url="http://x", chunk_chars=1200, overlap=120)
    assert len(chunks) >= 2
    assert all(c.content_hash for c in chunks)
    assert all(c.url == "http://x" for c in chunks)


def test_hybrid_retriever_bm25_only():
    chunks = chunk_text("Python 3.13 引入了 free-threading，PEP 703 移除 GIL。" * 5)
    chunks += chunk_text("LangGraph 是状态机编排框架。" * 5)
    r = HybridRetriever(chunks)
    hits = r.retrieve("PEP 703 GIL", top_k=5)
    assert hits
    assert "703" in hits[0][0].text


class _StubVectorIndex:
    """返回固定排名，用于验证向量分支确实被并入 RRF（不依赖真实 embedding）。"""

    def __init__(self, hits: list[tuple[int, float]]) -> None:
        self._hits = hits

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        return self._hits[:top_k]


def test_hybrid_retriever_vector_branch_when_bm25_misses():
    # 查询词在语料里完全不存在 → BM25 无命中，只有向量分支能召回
    chunks = chunk_text("内容甲。" * 30)
    r = HybridRetriever(chunks, _StubVectorIndex([(0, 0.9)]))
    hits = r.retrieve("zzz-not-exist", top_k=5)
    assert hits and hits[0][0] is chunks[0]


def _settings(**kw) -> Settings:
    base = {"rag_enabled": True, "rag_use_vector": False, "rag_use_reranker": False, "rag_top_k": 4}
    base.update(kw)
    return Settings(**base)


def test_evidence_selects_relevant_chunk_and_keeps_citation_id():
    """整页灌入 → 选块：只留相关段落，且 [citation_id] 仍指回整页快照。"""
    noise = "无关噪声。" * 400
    sources = [
        {"cid": 11, "url": "http://a", "title": "A", "text": noise + "PEP 703 移除了 GIL 限制。"},
        {"cid": 22, "url": "http://b", "title": "B", "text": "LangGraph 是状态机编排框架。" * 200},
    ]
    blocks = _build_evidence("PEP 703 GIL", sources, _settings())
    joined = "\n".join(blocks)
    assert "citation_id=11" in joined
    assert "703" in joined
    assert len(joined) < len(sources[0]["text"])  # 未整页灌入


def test_evidence_disabled_returns_empty():
    sources = [{"cid": 1, "url": "u", "title": "t", "text": "内容" * 50}]
    assert _build_evidence("q", sources, _settings(rag_enabled=False)) == []


def test_fallback_evidence_truncates_per_source():
    settings = _settings(researcher_max_tokens_per_source=10)  # 40 字符上限
    sources = [{"cid": 7, "url": "u", "title": "t", "text": "字" * 200}]
    blocks = _fallback_evidence(sources, settings)
    assert len(blocks) == 1
    assert "[citation_id=7]" in blocks[0]
    assert "[truncated]" in blocks[0]
