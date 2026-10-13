"""查询改写：子问题 → 检索式。用 light 模型，失败时原样返回（幂等安全）。"""
from __future__ import annotations

_REWRITE_PROMPT = """将以下研究子问题改写成 2-3 个适合搜索引擎的查询词组。
要求：每个查询一行，不含序号；保留专有名词原文；必要时补充英文同义词。

子问题：{question}"""


def build_rewrite_messages(question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "你是检索式改写器，只输出查询词组。"},
        {"role": "user", "content": _REWRITE_PROMPT.format(question=question)},
    ]


def parse_rewritten(text: str, fallback: str) -> list[str]:
    lines = [line.strip("- •0123456789.、 ") for line in text.splitlines()]
    queries = [line for line in lines if line and len(line) >= 2]
    return queries[:3] or [fallback]
