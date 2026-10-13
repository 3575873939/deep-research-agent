"""synthesizer 节点：findings → 结构化报告草稿。"""
from __future__ import annotations

from ..config import Settings
from ..provider.base import MeteredLLM
from .prompts import SYNTHESIZER_SYSTEM
from .state import ResearchState


async def synthesizer_node(state: ResearchState, *, llm: MeteredLLM, settings: Settings) -> dict:
    findings_text = []
    for i, f in enumerate(state.get("findings", []), 1):
        tag = "（来源曾检出注入，谨慎采信）" if f.get("injected") else ""
        findings_text.append(f"### 子问题 {i}：{f['subquestion']}{tag}\n{f['summary']}")

    budget_note = ""
    if state.get("budget_exhausted"):
        budget_note = "\n\n注意：本次运行预算截断，报告需在开头标注「预算截断版，调研不完整」。"

    result = await llm.chat(
        [
            {"role": "system", "content": SYNTHESIZER_SYSTEM},
            {"role": "user", "content": f"用户问题：{state['question']}\n\n调研材料：\n\n"
             + "\n\n".join(findings_text) + budget_note},
        ],
        model=settings.model_for("synthesizer"),
        node="synthesizer",
        temperature=0.3,
        max_tokens=6000,
    )
    return {"draft": result.content}
