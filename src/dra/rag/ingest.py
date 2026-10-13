"""文档切分入库：~500 token / 50 overlap，保留 URL + 原文哈希（溯源用）。

中文场景按字符近似 token（1 token ≈ 1.5 中文字 / 4 英文字符），取保守近似。
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

CHUNK_CHARS = 1200  # ≈ 500 token 的中文
OVERLAP_CHARS = 120  # ≈ 50 token


@dataclass
class Chunk:
    text: str
    url: str = ""
    title: str = ""
    idx: int = 0
    content_hash: str = ""
    meta: dict = field(default_factory=dict)


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def chunk_text(text: str, *, url: str = "", title: str = "",
               chunk_chars: int = CHUNK_CHARS, overlap: int = OVERLAP_CHARS) -> list[Chunk]:
    """滑窗切分。窗口在句子边界处对齐，减少断句。"""
    if not text:
        return []
    chunks: list[Chunk] = []
    start, idx = 0, 0
    while start < len(text):
        end = min(start + chunk_chars, len(text))
        if end < len(text):
            # 尽量在句号/换行处断开
            boundary = max(text.rfind("。", start, end), text.rfind("\n", start, end))
            if boundary > start + chunk_chars // 2:
                end = boundary + 1
        piece = text[start:end].strip()
        if piece:
            chunks.append(Chunk(
                text=piece, url=url, title=title, idx=idx, content_hash=hash_text(piece),
            ))
            idx += 1
        start = max(end - overlap, start + 1)
        if end >= len(text):
            break
    return chunks
