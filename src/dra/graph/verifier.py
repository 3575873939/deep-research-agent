"""verifier 节点：独立核查每条带 [cite:N] 结论是否被引用原文支持。

核心设计（面试必讲）：
1. verifier 是独立节点，评估与生成分离，可独立测试
2. 默认用与生成侧不同的模型（节点级路由），避免"自我认可"偏差
3. 规则前置：先抽取 [cite:N] 对，无引用的结论直接记 unverified 不调 LLM
"""
from __future__ import annotations

import re

from ..config import Settings
from ..provider.base import MeteredLLM
from ..tools.citations import CitationStore
from ..trace.logger import RunTrace
from .prompts import VERIFIER_SYSTEM
from .state import ResearchState, Verification

# 结论切分：按句子切，含 [cite:N] 的句子视为一条待核结论
_SENT_SPLIT = re.compile(r"(?<=[。！？.!?\n])")
_CITE = re.compile(r"\[cite:(\d+)\]")


def extract_claims(draft: str) -> list[tuple[str, int | None]]:
    """抽取 (结论, citation_id)。一条句子多个引用时取第一个（逐条核查太费 token）。"""
    claims: list[tuple[str, int | None]] = []
    for sent in _SENT_SPLIT.split(draft):
        sent = sent.strip()
        if len(sent) < 6 or sent.startswith("#"):
            continue
        m = _CITE.search(sent)
        claims.append((sent, int(m.group(1)) if m else None))
    return claims


async def verifier_node(
    state: ResearchState,
    *,
    llm: MeteredLLM,
    settings: Settings,
    citations: CitationStore,
    trace: RunTrace,
) -> dict:
    draft = state.get("draft", "")
    report: list[Verification] = []

    for claim, cid in extract_claims(draft):
        if cid is None:
            report.append(Verification(claim=claim, citation_id=None, verdict="unverified"))
            continue
        cite = citations.get(cid)
        if cite is None:
            report.append(Verification(claim=claim, citation_id=cid, verdict="unsupported"))
            continue

        result = await llm.chat(
            [
                {"role": "system", "content": VERIFIER_SYSTEM},
                {"role": "user", "content": f"结论：{claim}\n\n引用原文（快照哈希 {cite.content_hash[:8]}）：\n{cite.snapshot[:3000]}"},
            ],
            model=settings.model_for("verifier"),
            node="verifier",
            temperature=0.0,
            max_tokens=1024,  # 推理模型的思考链会吃 token，给足余量
        )
        verdict = _parse_verdict(result.content)
        report.append(Verification(claim=claim, citation_id=cid, verdict=verdict))

    # 写进 trace，evals 直接消费
    for v in report:
        key = f"verify_{v['verdict']}"
        if key in ("verify_supported", "verify_partial", "verify_unsupported"):
            setattr(trace, key, getattr(trace, key) + 1)
    return {"verify_report": report, "verify_round": state.get("verify_round", 0) + 1}


def _parse_verdict(text: str) -> str:
    first = text.strip().splitlines()[0].lower() if text.strip() else ""
    for v in ("supported", "partial", "unsupported"):
        if v in first:
            return v
    return "unsupported"  # 判定输出异常时从严


def should_redo(state: ResearchState, settings: Settings) -> str:
    """路由：unsupported 比例超阈值且未达轮次上限 → 打回重查；否则结束。"""
    report = state.get("verify_report", [])
    if not report:
        return "finalize"
    bad = sum(1 for v in report if v["verdict"] == "unsupported")
    if bad > len(report) * 0.2 and state.get("verify_round", 0) < settings.verifier_max_rounds:
        return "redo"
    return "finalize"
