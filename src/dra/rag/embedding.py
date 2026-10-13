"""向量索引（可选增强）：sentence-transformers + bge-small-zh。

设计同 reranker：向量不是必需依赖，任何环节缺失都返回 None，调用方退化为 BM25-only
——保证 CI / 无 GPU / 断网环境下混合检索全链路仍可跑。
研究 Agent 每 run 语料量小（几十页），进程内暴力余弦检索足够，不引入向量数据库。
"""
from __future__ import annotations

from .ingest import Chunk

# 进程级模型缓存：一个 run 有多个子问题并行，若每个子问题各加载一次模型会重复
# 付出十几秒加载 + 数倍内存（且加载是同步阻塞，会串行拖慢整个事件循环）。
_MODEL_CACHE: dict[tuple[str, str, bool], object] = {}


def _get_model(model_name: str, device: str, local_files_only: bool):
    key = (model_name, device, local_files_only)
    model = _MODEL_CACHE.get(key)
    if model is None:
        from sentence_transformers import SentenceTransformer  # 可选依赖

        model = SentenceTransformer(
            model_name, device=device, local_files_only=local_files_only
        )
        _MODEL_CACHE[key] = model
    return model


class SentenceTransformerIndex:
    """进程内向量索引：归一化 embedding 后的矩阵点积 = 余弦相似度。

    实现 retriever.VectorIndex 协议：search(query, top_k) -> [(chunk_idx, score)]
    """

    def __init__(
        self,
        chunks: list[Chunk],
        *,
        model_name: str = "BAAI/bge-small-zh-v1.5",
        device: str = "cpu",
        batch_size: int = 32,
        local_files_only: bool = True,
        min_score: float = 0.0,
    ) -> None:
        # local_files_only 默认 True：只读本地缓存，避免联网检查元数据时长时间阻塞
        # （实测在无法直连 huggingface.co 的网络下会挂起数分钟，违背"不阻断主链路"的设计）
        self._model = _get_model(model_name, device, local_files_only)
        self._min_score = min_score
        self._embs = self._model.encode(
            [c.text for c in chunks],
            normalize_embeddings=True,
            batch_size=batch_size,
        )

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        import numpy as np

        q = np.asarray(self._model.encode([query], normalize_embeddings=True)[0])
        sims = np.asarray(self._embs) @ q
        order = sims.argsort()[::-1]
        # 阈值过滤：向量召回无天然"零命中"，不设阈值会把无关块灌进证据（实测有子任务
        # 因此拿到完全不相关的材料、最终零引用）。留意 bge 相关对通常在 0.4 以上。
        hits = [(int(i), float(sims[i])) for i in order if float(sims[i]) >= self._min_score]
        return hits[:top_k]


def build_vector_index(
    chunks: list[Chunk],
    *,
    model_name: str = "BAAI/bge-small-zh-v1.5",
    device: str = "cpu",
    local_files_only: bool = True,
    min_score: float = 0.0,
) -> SentenceTransformerIndex | None:
    """构建向量索引；缺失依赖 / 模型未缓存 / 空语料 一律返回 None（降级是设计）。"""
    if not chunks:
        return None
    try:
        return SentenceTransformerIndex(
            chunks, model_name=model_name, device=device,
            local_files_only=local_files_only, min_score=min_score,
        )
    except Exception:  # noqa: BLE001 - 降级不阻断主链路
        return None