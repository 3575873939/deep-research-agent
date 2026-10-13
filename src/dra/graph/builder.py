"""graph 组装 + interrupt 配置 + 运行入口（同步包装，CLI/evals/server 共用）。"""
from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from langgraph.graph import END, StateGraph
from langgraph.types import Command

from ..config import Settings, get_settings
from ..guard.budget import BudgetExceeded, BudgetGuard
from ..provider.base import CircuitBreaker, MeteredLLM
from ..provider.modelscope import OpenAICompatibleProvider
from ..tools.base import Tool
from ..tools.browser import BrowserFetch
from ..tools.citations import CitationStore
from ..tools.mcp_client import open_toolbox, parse_mcp_servers
from ..tools.search import make_search_tool
from ..trace.logger import RunTrace, TraceLogger
from .human import human_review_node, route_after_human, route_after_verify
from .planner import planner_node
from .researcher import researcher_node
from .state import ResearchState
from .synthesizer import synthesizer_node
from .verifier import verifier_node


def extract_interrupt(final: dict[str, Any]) -> dict[str, Any] | None:
    """从 graph 返回的 state 里取出待人工确认的 payload（无则返回 None）。

    LangGraph 遇到 interrupt() 会给 state 加 `__interrupt__` 键，值为 Interrupt 对象元组。
    """
    raw = final.get("__interrupt__")
    if not raw:
        return None
    first = raw[0] if isinstance(raw, (list, tuple)) else raw
    value = getattr(first, "value", None)
    if value is None:
        value = first if isinstance(first, dict) else str(first)
    return {"id": getattr(first, "id", None), "value": value}


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

    @asynccontextmanager
    async def _checkpointer(self, settings: Settings):
        """人工介入需要 checkpointer 才能断点恢复；开关关闭时不引入任何开销。"""
        if not settings.dra_human_review:
            yield None
            return
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        Path(settings.dra_checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        async with AsyncSqliteSaver.from_conn_string(settings.dra_checkpoint_path) as cp:
            yield cp

    @staticmethod
    def _thread_config(run_id: str, enabled: bool) -> dict[str, Any]:
        """启用 checkpointer 时必须给 thread_id，否则无法定位/恢复该次运行。"""
        return {"configurable": {"thread_id": run_id}} if enabled else {}

    @asynccontextmanager
    async def _mcp_search(self, settings: Settings):
        """若配置了 MCP 检索工具，连接对应 server 并产出该工具；否则产出 None。

        默认不配置 → 不启动任何子进程、零额外开销。server 连不上或工具名不存在
        一律产出 None，由调用方降级到内置检索源。
        """
        configs = parse_mcp_servers(settings.dra_mcp_servers)
        wanted = settings.dra_mcp_search_tool.strip()
        if not configs or not wanted:
            yield None
            return
        try:
            import mcp  # noqa: F401 - 可选依赖，缺失则降级
        except ImportError:
            print("[mcp] 未安装 mcp 包，降级到内置检索源（pip install mcp）")
            yield None
            return
        async with open_toolbox(configs) as tools:
            tool = tools.get(wanted)
            if tool is None:
                print(f"[mcp] 未找到检索工具 '{wanted}'，降级到内置检索源")
            yield tool

    def build_graph(self, budget: BudgetGuard, trace: RunTrace, settings: Settings | None = None,
                    provider: Any | None = None, checkpointer: Any | None = None,
                    search: Tool | None = None) -> Any:
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
        search_tool = search or self.search   # MCP 检索工具优先（若配置），否则内置
        g.add_node("researcher", safe(lambda st: researcher_node(
            st, llm=llm, settings=s, search=search_tool, fetch=self.fetch,
            citations=self.citations, trace=trace)))
        g.add_node("synthesizer", safe(lambda st: synthesizer_node(st, llm=llm, settings=s)))
        g.add_node("verifier", safe(lambda st: verifier_node(
            st, llm=llm, settings=s, citations=self.citations, trace=trace)))
        # 人工确认节点：开关关闭或无风险结论时原样放行（返回 auto），不影响既有流程
        g.add_node("human_review", safe(lambda st: human_review_node(st, settings=s)))

        g.set_entry_point("planner")
        g.add_edge("planner", "researcher")
        g.add_edge("researcher", "synthesizer")
        g.add_edge("synthesizer", "verifier")
        g.add_conditional_edges(
            "verifier",
            lambda st: route_after_verify(st, s),
            {"redo": "researcher", "review": "human_review", "finalize": END},
        )
        g.add_conditional_edges(
            "human_review",
            lambda st: route_after_human(st, s),
            {"redo": "researcher", "finalize": END},
        )
        return g.compile(checkpointer=checkpointer)

    async def run(self, question: str, *, model_overrides: dict[str, str] | None = None,
                  provider: Any | None = None) -> tuple[ResearchState, RunTrace]:
        settings = self.settings.model_copy(update=model_overrides) if model_overrides else self.settings
        trace = RunTrace(run_id=uuid.uuid4().hex[:12], question=question)
        budget = BudgetGuard(settings.dra_max_tokens_per_run)

        initial: ResearchState = {"question": question, "run_id": trace.run_id, "messages": []}
        async with self._mcp_search(settings) as mcp_search, self._checkpointer(settings) as cp:
            graph = self.build_graph(budget, trace, settings=settings, provider=provider,
                                     checkpointer=cp, search=mcp_search)
            try:
                final = await graph.ainvoke(initial, self._thread_config(trace.run_id, bool(cp)))
            except BudgetExceeded:
                final = {**initial, "final_report": "[预算耗尽] 无法完成调研", "budget_exhausted": True}
        trace.budget_exhausted = bool(final.get("budget_exhausted"))
        trace.subquestions = final.get("subquestions", [])
        self.traces.finish(trace)
        return final, trace

    async def resume(self, run_id: str, decision: dict[str, Any], *,
                     model_overrides: dict[str, str] | None = None,
                     provider: Any | None = None) -> tuple[ResearchState, RunTrace]:
        """从人工确认断点恢复运行。

        decision 形如 {"action": "accept"} 或 {"action": "redo"}；run_id 必须是
        原先触发 interrupt 的那次运行（checkpointer 按 thread_id=run_id 存放状态）。
        """
        settings = self.settings.model_copy(update=model_overrides) if model_overrides else self.settings
        if not settings.dra_human_review:
            raise RuntimeError("恢复运行需要 DRA_HUMAN_REVIEW=true（否则没有 checkpointer）")

        trace = RunTrace(run_id=run_id, question="(resumed)")
        budget = BudgetGuard(settings.dra_max_tokens_per_run)
        config = self._thread_config(run_id, True)

        async with self._mcp_search(settings) as mcp_search, self._checkpointer(settings) as cp:
            graph = self.build_graph(budget, trace, settings=settings, provider=provider,
                                     checkpointer=cp, search=mcp_search)
            snapshot = await graph.aget_state(config)
            if not (snapshot.values or {}):
                raise RuntimeError(f"未找到 run_id={run_id} 的断点状态")
            trace.question = (snapshot.values or {}).get("question", "(resumed)")
            try:
                final = await graph.ainvoke(Command(resume=decision), config)
            except BudgetExceeded:
                final = {**snapshot.values, "final_report": "[预算耗尽] 无法完成调研",
                         "budget_exhausted": True}

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
        initial: ResearchState = {"question": question, "run_id": trace.run_id, "messages": []}

        yield {"type": "plan", "data": {"run_id": trace.run_id, "question": question}}
        final_report = ""
        paused = False
        try:
            print(f"[run_stream] 开始 graph，问题: {question[:50]}")
            async with self._mcp_search(settings) as mcp_search, self._checkpointer(settings) as cp:
                graph = self.build_graph(budget, trace, settings=settings, provider=provider,
                                         checkpointer=cp, search=mcp_search)
                config = self._thread_config(trace.run_id, bool(cp))
                async for event in graph.astream(initial, config, stream_mode="updates"):
                    for node, update in event.items():
                        # 人工确认断点：把待确认内容发给前端，等待 /research/resume
                        if node == "__interrupt__":
                            paused = True
                            payload = extract_interrupt({"__interrupt__": update}) or {}
                            yield {"type": "interrupt", "data": {
                                "run_id": trace.run_id, **payload}}
                            continue
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
                                            "paused": paused,
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
