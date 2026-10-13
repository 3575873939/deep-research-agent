"""planner 节点：问题 → 子问题列表。"""
from __future__ import annotations

import json
import re

from ..config import Settings
from ..provider.base import MeteredLLM
from .prompts import PLANNER_SYSTEM
from .state import ResearchState, ResearchTask


def parse_subquestions(text: str, fallback_question: str) -> list[str]:
    """从模型输出提取 JSON。模型可能裹 markdown 围栏，宽容解析。"""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(0))
            subs = [s.strip() for s in data.get("subquestions", []) if isinstance(s, str) and s.strip()]
            if subs:
                return subs[:5]
        except json.JSONDecodeError:
            pass
    # 兜底：按行拆
    lines = [re.sub(r"^[-*\d.、\s]+", "", line).strip() for line in text.splitlines()]
    subs = [line for line in lines if len(line) >= 6][:5]
    return subs or [fallback_question]


def parse_tasks(text: str, fallback_question: str) -> list[ResearchTask]:
    """从模型输出提取任务级规划 (title/intent/query)。宽容解析 JSON。"""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    tasks: list[ResearchTask] = []
    if m:
        try:
            data = json.loads(m.group(0))
            raw = data.get("tasks") or data.get("subquestions") or []
            if isinstance(raw, dict):          # {"tasks": {...}} 偶发单对象
                raw = [raw]
            for i, item in enumerate(raw, 1):
                if isinstance(item, str):
                    # 兼容纯子问题字符串：降级为单字段任务
                    tasks.append({"id": i, "title": item, "intent": "", "query": item, "status": "pending"})
                elif isinstance(item, dict):
                    query = str(item.get("query") or item.get("title") or "").strip()
                    if not query:
                        continue
                    tasks.append({
                        "id": i,
                        "title": str(item.get("title") or f"任务{i}").strip(),
                        "intent": str(item.get("intent") or "聚焦主题的关键问题").strip(),
                        "query": query,
                        "status": "pending",
                    })
        except json.JSONDecodeError:
            pass
    if tasks:
        return tasks[:5]
    # 兜底：退化为单个任务（等价于原 fallback）
    return [{
        "id": 1, "title": "综合调研", "intent": "全面收集主题的背景与最新动态",
        "query": fallback_question, "status": "pending",
    }]


async def planner_node(state: ResearchState, *, llm: MeteredLLM, settings: Settings) -> dict:
    result = await llm.chat(
        [
            {"role": "system", "content": PLANNER_SYSTEM},
            {"role": "user", "content": state["question"]},
        ],
        model=settings.model_for("planner"),
        node="planner",
        temperature=0.2,
    )
    tasks = parse_tasks(result.content, state["question"])
    # 下游 researcher 需要子问题列表：取自 task.query（兼容原 subquestions 通路）
    subs = [t["query"] for t in tasks]
    return {"tasks": tasks, "subquestions": subs}
