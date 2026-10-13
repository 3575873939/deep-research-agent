"""human_review 节点：高风险动作人工确认（LangGraph 原生 interrupt + checkpointer）。

设计要点（面试讲点）：
1. **不是每一步都打断人**——只在 verifier 检出「缺乏引用支持」的结论时才暂停，
   人工介入是例外路径而非常规流程。
2. **用 LangGraph 原生 interrupt()**，不自造轮子：状态由 checkpointer 持久化，
   进程重启后仍可从断点恢复（`Command(resume=...)`）。
3. **默认关闭**（`dra_human_review=False`）：评测与批量跑必须全自动，
   人为确认会破坏可复现性——这是刻意的默认值，不是遗漏。
"""
from __future__ import annotations

from typing import Any

from langgraph.types import interrupt

from ..config import Settings
from .state import ResearchState
from .verifier import should_redo

# 送给人看的结论片段上限，避免 payload 过大
_MAX_CLAIMS_IN_PAYLOAD = 20


def risky_claims(state: ResearchState) -> list[dict[str, Any]]:
    """挑出非 supported 的结论，作为人工判断的输入。"""
    return [
        {"claim": v["claim"][:300], "citation_id": v["citation_id"], "verdict": v["verdict"]}
        for v in state.get("verify_report", [])
        if v["verdict"] != "supported"
    ]


def _normalize_decision(raw: Any) -> str:
    """把人工返回值归一化为 accept / redo。"""
    if isinstance(raw, str):
        return "redo" if raw.strip().lower() == "redo" else "accept"
    if isinstance(raw, dict):
        val = str(raw.get("action") or raw.get("value") or "accept").strip().lower()
        return "redo" if val == "redo" else "accept"
    return "accept"


async def human_review_node(state: ResearchState, *, settings: Settings) -> dict:
    """在报告定稿前插入人工确认点。

    返回 {"human_decision": ...}：
    - auto：未启用开关，或没有需要人看的风险结论 → 不打断
    - accept / redo：人工选择的结果
    """
    if not settings.dra_human_review:
        return {"human_decision": "auto"}

    risky = risky_claims(state)
    if len(risky) < settings.human_review_min_risky:
        return {"human_decision": "auto"}

    decision = interrupt({
        "kind": "report_approval",
        "question": state.get("question", ""),
        "message": (
            f"检测到 {len(risky)} 条结论缺乏完整的引用支持。"
            "接受当前报告，还是打回重新调研？"
        ),
        "risky_count": len(risky),
        "risky_claims": risky[:_MAX_CLAIMS_IN_PAYLOAD],
        "options": [
            {"value": "accept", "label": "接受当前报告"},
            {"value": "redo", "label": "打回重新调研"},
        ],
    })
    return {"human_decision": _normalize_decision(decision)}


def route_after_verify(state: ResearchState, settings: Settings) -> str:
    """verifier 之后的分支：先按质量自动打回，再看是否要人工确认。"""
    if should_redo(state, settings) == "redo":
        return "redo"
    if settings.dra_human_review:
        return "review"
    return "finalize"


def route_after_human(state: ResearchState, settings: Settings) -> str:
    """人工确认之后：选择 redo 且未达轮次上限才回调研，否则定稿。"""
    if (state.get("human_decision") == "redo"
            and state.get("verify_round", 0) < settings.verifier_max_rounds):
        return "redo"
    return "finalize"
