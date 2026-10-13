"""graph 组装 + interrupt 配置 + 运行入口（同步包装，CLI/evals/server 共用）。"""
from __future__ import annotations

import asyncio
import uuid
from typing import Any, AsyncIterator

from langgraph.graph import END, StateGraph

from ..config import Settings, get_settings
from ..guard.budget import BudgetExceeded, BudgetGuard
from ..provider.base import CircuitBreaker, MeteredLLM
from ..provider.modelscope import OpenAICompatibleProvider
from ..tools.base import Tool
from ..tools.browser import BrowserFetch
from ..tools.citations import CitationStore
from ..tools.search import make_search_tool
from ..trace.logger import RunTrace, TraceLogger
from .planner import planner_node
from .researcher import researcher_node
from .state import ResearchState
from .synthesizer import synthesizer_node
from .verifier import should_redo, verifier_node


class ResearchRuntime:
    """持有全部资源的运行时：provider/tools/citations/trace。一个进程一份。"""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        provider = OpenAICompatibleProvider(
            self.settings.dra_base_url, self.settings.dra_api_key,
            timeout_s=self.settings.llm_timeout_s,
            thinking=self.settings.dra_thinking_enabled,
        )
        self.breaker = CircuitBreaker(self.settings.dra_rate_limit_cooldown_s)
        self._provider = provider
        self.search: Tool = make_search_tool(
            self.settings.tavily_api_key, self.settings.serpapi_key
        )
        self.fetch: Tool = BrowserFetch()
        self.citations = CitationStore(self.settings.dra_db_path)
        self.traces = TraceLogger(self.settings.dra_trace_path)

    def _budget_safe(self, budget: BudgetGuard):
        """把 BudgetExceeded 降级为 state 标记，graph 继续走 synthesize（用已有材料成稿）。"""
        def deco(fn):
            async def wrapped(state, **kwargs):
                try:
                    return await fn(state, **kwargs)
                except BudgetExceeded:
                    budget_exhausted_note = {"budget_exhausted": True}
                    return budget_exhausted_note
            return wrapped
        return deco

    def build_graph(self, budget: BudgetGuard, trace: RunTrace, settings: Settings | None = None,
                    provider: Any | None = None) -> Any:
        s = settings or self.settings
        llm = MeteredLLM(
            provider or self._provider, budget, self.breaker,
            on_usage=lambda node, u: (
                trace.token_usage.__setitem__(node, trace.token_usage.get(node, 0) + u.total),
                setattr(trace, "llm_calls", trace.llm_calls + 1),
            ),
            min_interval_s=s.dra_min_call_interval_s,
        )
        safe = self._budget_safe(budget)

        g = StateGraph(ResearchState)
        g.add_node("planner", safe(lambda st: planner_node(st, llm=llm, settings=s)))
        g.add_node("researcher", safe(lambda st: researcher_node(
            st, llm=llm, settings=s, search=self.search, fetch=self.fetch,
            citations=self.citations, trace=trace)))
        g.add_node("synthesizer", safe(lambda st: synthesizer_node(st, llm=llm, settings=s)))
        g.add_node("verifier", safe(lambda st: verifier_node(
            st, llm=llm, settings=s, citations=self.citations, trace=trace)))

        g.set_entry_point("planner")
        g.add_edge("planner", "researcher")
        g.add_edge("researcher", "synthesizer")
        g.add_edge("synthesizer", "verifier")
        g.add_conditional_edges(
            "verifier",
            lambda st: should_redo(st, s),
            {"redo": "researcher", "finalize": END},
        )
        return g.compile()

    async def run(self, question: str, *, model_overrides: dict[str, str] | None = None,
                  provider: Any | None = None) -> tuple[ResearchState, RunTrace]:
        settings = self.settings.model_copy(update=model_overrides) if model_overrides else self.settings
        trace = RunTrace(run_id=uuid.uuid4().hex[:12], question=question)
        budget = BudgetGuard(settings.dra_max_tokens_per_run)
        graph = self.build_graph(budget, trace, settings=settings, provider=provider)

        initial: ResearchState = {"question": question, "run_id": trace.run_id, "messages": []}
        try:
            final = await graph.ainvoke(initial)
        except BudgetExceeded:
            final = {**initial, "final_report": "[预算耗尽] 无法完成调研", "budget_exhausted": True}
        trace.budget_exhausted = bool(final.get("budget_exhausted"))
        trace.subquestions = final.get("subquestions", [])
        self.traces.finish(trace)
        return final, trace

    async def run_stream(self, question: str, *, model_overrides: dict[str, str] | None = None,
                         provider: Any | None = None) -> AsyncIterator[dict[str, Any]]:
        """SSE 事件源：按节点粒度发 plan/tool_call/token/verify/done 事件。"""
        settings = self.settings.model_copy(update=model_overrides) if model_overrides else self.settings
        trace = RunTrace(run_id=uuid.uuid4().hex[:12], question=question)
        budget = BudgetGuard(settings.dra_max_tokens_per_run)
        graph = self.build_graph(budget, trace, settings=settings, provider=provider)
        initial: ResearchState = {"question": question, "run_id": trace.run_id, "messages": []}

        yield {"type": "plan", "data": {"run_id": trace.run_id, "question": question}}
        final_report = ""
        try:
            print(f"[run_stream] 开始 graph，问题: {question[:50]}")
            async for event in graph.astream(initial, stream_mode="updates"):
                for node, update in event.items():
                    if node == "planner" and update.get("subquestions"):
                        yield {"type": "plan", "data": {"subquestions": update["subquestions"]}}
                    elif node == "researcher":
                        yield {"type": "tool_result", "data": {
                            "findings_so_far": len(update.get("findings", [])),
                            "tool_calls": len(trace.tool_calls),
                        }}
                        # 任务级进度事件（展示层）
                        for t in update.get("tasks", []):
                            yield {"type": "task_status", "data": {
                                "task_id": t["id"],
                                "title": t.get("title", ""),
                                "intent": t.get("intent", ""),
                                "status": t.get("status", "pending"),
                                "summary": t.get("summary", ""),
                                "sources": t.get("sources", []),
                                "citation_ids": t.get("citation_ids", []),
                            }}
                    elif node == "synthesizer":
                        if update.get("draft"):
                            final_report = update["draft"]
                        yield {"type": "token", "data": {"draft_chars": len(update.get("draft", ""))}}
                    elif node == "verifier":
                        rep = update.get("verify_report", [])
                        yield {"type": "verify", "data": {
                            "supported": sum(1 for v in rep if v["verdict"] == "supported"),
                            "partial": sum(1 for v in rep if v["verdict"] == "partial"),
                            "unsupported": sum(1 for v in rep if v["verdict"] == "unsupported"),
                        }}
            yield {"type": "done", "data": {"run_id": trace.run_id,
                                            "tokens": trace.total_tokens,
                                            "budget_exhausted": trace.budget_exhausted,
                                            "report": final_report}}
        except BudgetExceeded:
            trace.budget_exhausted = True
            yield {"type": "error", "data": {"error": "budget_exhausted"}}
        except Exception as e:
            print(f"[run_stream] 异常: {type(e).__name__}: {e}")
            yield {"type": "error", "data": {"error": f"{type(e).__name__}: {e}"}}
        finally:
            self.traces.finish(trace)


def run_question(question: str, settings: Settings | None = None) -> tuple[ResearchState, RunTrace]:
    """同步入口：CLI / evals 用。"""
    return asyncio.run(ResearchRuntime(settings).run(question))
