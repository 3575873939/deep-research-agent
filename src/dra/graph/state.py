"""LangGraph 状态定义。"""
from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages


class Finding(TypedDict, total=False):
    subquestion: str
    summary: str
    citation_ids: list[int]
    injected: bool   # 该 finding 来源中检出过注入
    urls: list[str]  # 该 finding 的来源 URL（展示层用）


class ResearchTask(TypedDict, total=False):
    """任务级状态（展示/编排层），与 findings 并行，不参与引用核查链。"""
    id: int
    title: str              # 任务名（规划器生成）
    intent: str             # 任务要解决的核心问题
    query: str              # 检索查询（researcher 的调研单元）
    status: str             # pending / in_progress / completed / failed / skipped
    summary: str            # 该任务总结（含 [cite:N]）
    sources: list[str]      # 来源 URL 概览
    citation_ids: list[int] # 该任务用到的引用 id


class Verification(TypedDict):
    claim: str
    citation_id: int | None
    verdict: str  # supported / partial / unsupported


class ResearchState(TypedDict, total=False):
    question: str
    run_id: str
    subquestions: list[str]
    tasks: list[ResearchTask]   # 任务级状态（规划器产出，展示层）
    findings: list[Finding]
    draft: str
    verify_report: list[Verification]
    verify_round: int            # 打回重查轮次，上限 verifier_max_rounds
    budget_exhausted: bool
    final_report: str
    error: str
    # SSE 事件队列由 runtime 注入，不进 state（checkpoint 序列化安全）
    messages: Annotated[list[Any], add_messages]
