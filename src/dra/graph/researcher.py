"""researcher 节点：对每个子问题执行 search → fetch → 摘要 → 引用入库。

纪律（写死，不靠 prompt 自觉）：
- 工具调用循环上限 researcher_max_steps
- 同一子问题最多重查 researcher_max_retry_per_question 次
- 网页文本进 context 前注入清洗 + 隔离包裹
- 引用快照立即入库，[cite:N] 对应 SQLite 行 id
"""
from __future__ import annotations

import asyncio
import json
import re

from ..config import Settings
from ..guard.injection import sanitize_tool_output, wrap_untrusted
from ..provider.base import MeteredLLM
from ..rag import HybridRetriever, Reranker, build_vector_index, chunk_text
from ..tools.base import Tool, call_tool
from ..tools.citations import CitationStore
from ..trace.logger import RunTrace
from .prompts import RESEARCHER_SYSTEM
from .state import Finding, ResearchState, ResearchTask


def _expand_queries(subquestion: str) -> list[str]:
    """规则扩展搜索 query：官方文档优先、英文精确术语、中文 broad 搜索。零成本确定性。"""
    base = subquestion.strip()
    queries = []

    # 官方文档优先：技术问题优先搜官方站（site 限定强制召回）
    tech_map = {
        "langgraph": ["site:langchain-ai.github.io/langgraph", "site:github.com/langchain-ai/langgraph"],
        "langchain": ["site:python.langchain.com", "site:github.com/langchain-ai/langchain"],
        "python": ["site:docs.python.org", "site:github.com/python/cpython"],
        "fastapi": ["site:fastapi.tiangolo.com", "site:github.com/tiangolo/fastapi"],
        "react": ["site:react.dev", "site:github.com/facebook/react"],
        "vue": ["site:vuejs.org", "site:github.com/vuejs/core"],
        "django": ["site:docs.djangoproject.com", "site:github.com/django/django"],
        "flask": ["site:flask.palletsprojects.com", "site:github.com/pallets/flask"],
        "pytorch": ["site:pytorch.org", "site:github.com/pytorch/pytorch"],
        "tensorflow": ["site:tensorflow.org", "site:github.com/tensorflow/tensorflow"],
        "transformers": ["site:huggingface.co/docs/transformers", "site:github.com/huggingface/transformers"],
        "bm25": ["en.wikipedia.org/wiki/Okapi_BM25", "site:github.com"],
        "mcp": ["site:modelcontextprotocol.io", "site:github.com/modelcontextprotocol"],
    }
    for tech, sites in tech_map.items():
        if tech in base.lower():
            for s in sites:
                queries.append(f"{s} {base}")
            break

    if not queries:
        # 概念解释类：优先英文 wiki + 中文技术社区
        if any(k in base for k in ["是什么", "原理", "机制", "架构", "设计", "实现"]):
            queries.append(f"{base} 官方文档")
            queries.append(f"{base} wikipedia")
        # 时效性问题：优先最新发布/更新日志
        elif any(k in base for k in ["最新", "2025", "2026", "版本", "发布", "更新", "价格", "费用"]):
            queries.append(f"{base} 2025 2026")
            queries.append(f"{base} changelog release notes")
        else:
            # 通用：中英双语 + 精确匹配
            queries.append(f"{base} 官方")
            queries.append(f'"{base}"')

    return queries[:3]


def _fallback_evidence(sources: list[dict], settings: Settings) -> list[str]:
    """降级路径：不做检索，整页截断后进 context（保留原行为，保证无 RAG 依赖时可用）。"""
    char_limit = settings.researcher_max_tokens_per_source * 4
    blocks = []
    for s in sources:
        text = s["text"]
        if len(text) > char_limit:
            text = text[:char_limit] + "... [truncated]"
        blocks.append(
            f"[citation_id={s['cid']}] {s['title']}\nURL: {s['url']}\n{wrap_untrusted(text)}"
        )
    return blocks


def _build_evidence(subquestion: str, sources: list[dict], settings: Settings) -> list[str]:
    """混合检索选证据块：切分 → BM25+向量双路召回 → RRF 融合 → 可选 rerank → 按来源聚合。

    为什么选块而不是整页灌入：抓回的网页大半与子问题无关，整页进 context 既烧 token
    又稀释注意力；检索式选块把 context 从"网页全文"收敛为"相关段落"。
    返回空列表表示检索不可用/无命中，调用方回退整页截断路径。
    """
    if not settings.rag_enabled:
        return []

    # 1) 切分：每块记录来源下标，保证引用仍指回整页快照（verifier 语义不变）
    chunks = []
    for si, s in enumerate(sources):
        for c in chunk_text(s["text"], url=s["url"], title=s["title"]):
            c.meta["src_idx"] = si
            chunks.append(c)
    if not chunks:
        return []

    # 2) 双路召回：向量索引可选，缺失时 HybridRetriever 自动只用 BM25
    vector_index = None
    if settings.rag_use_vector:
        vector_index = build_vector_index(
            chunks,
            model_name=settings.rag_embedding_model,
            local_files_only=settings.rag_local_files_only,
            min_score=settings.rag_vector_min_score,
        )
    retriever = HybridRetriever(chunks, vector_index)

    # 粗排多取，再按需 rerank 精排到 top_k
    hits = retriever.retrieve(subquestion, top_k=max(settings.rag_top_k * 3, settings.rag_top_k))
    if not hits:
        return []
    if settings.rag_use_reranker:
        hits = Reranker(settings.rag_reranker_model).rerank(
            subquestion, hits, top_k=settings.rag_top_k
        )
    else:
        hits = hits[: settings.rag_top_k]

    # 3) 按来源聚合：来源间按最佳名次排序，来源内多块拼接
    order: list[int] = []
    grouped: dict[int, list[str]] = {}
    for chunk, _score in hits:
        si = chunk.meta["src_idx"]
        if si not in grouped:
            grouped[si] = []
            order.append(si)
        if chunk.text not in grouped[si]:
            grouped[si].append(chunk.text)

    blocks = []
    for si in order:
        s = sources[si]
        body = "\n…\n".join(grouped[si])
        blocks.append(
            f"[citation_id={s['cid']}] {s['title']}\nURL: {s['url']}\n{wrap_untrusted(body)}"
        )
    return blocks


async def _research_one(
    subquestion: str,
    *,
    deps_llm: MeteredLLM,
    settings: Settings,
    search: Tool,
    fetch: Tool,
    citations: CitationStore,
    trace: RunTrace,
    run_id: str,
) -> Finding:
    cited_ids: list[int] = []
    evidence_blocks: list[str] = []
    injected_seen = False
    source_urls: list[str] = []

    # 1) 搜索：先规则扩展 query，再多 query 串行搜索合并去重（提升召回率）
    expanded = _expand_queries(subquestion)
    queries = [subquestion] + expanded

    search_data: list[dict] = []
    seen_urls: set[str] = set()
    for qi, q in enumerate(queries):
        if qi >= settings.researcher_max_retry_per_question + 1:
            break
        res = await call_tool(search, {"query": q}, timeout_s=settings.tool_timeout_s)
        trace.tool_calls.append({
            "tool": "search", "ok": res.ok, "error_type": res.error_type.value,
            "latency_s": round(res.latency_s, 2),
        })
        if res.ok:
            for item in res.data:
                u = item.get("url", "").strip()
                if u and u not in seen_urls:
                    seen_urls.add(u)
                    search_data.append(item)
        if len(search_data) >= settings.researcher_fetch_topk:
            break

    # 2) 并行抓取 topK 页面，快照入库
    async def _fetch_one(item: dict) -> dict:
        url = item.get("url", "")
        title = item.get("title", "")
        snippet = item.get("snippet", "")
        page_text = snippet
        page_injected = False
        if url:
            page = await call_tool(fetch, {"url": url}, timeout_s=settings.tool_timeout_s)
            trace.tool_calls.append({
                "tool": "fetch_page", "ok": page.ok, "error_type": page.error_type.value,
                "latency_s": round(page.latency_s, 2),
            })
            if page.ok:
                cleaned, injected = sanitize_tool_output(page.data["text"])
                page_injected = injected
                page_text = cleaned
        return {
            "url": url, "title": title, "text": page_text,
            "injected": page_injected,
        }

    fetched = await asyncio.gather(
        *[_fetch_one(item) for item in search_data[: settings.researcher_fetch_topk]]
    )
    # 2.5) 原文快照入库：每来源一条引用（快照=整页原文，verifier 据此核查，引用粒度不变）
    sources: list[dict] = []
    for f in fetched:
        injected_seen = injected_seen or f["injected"]
        source_urls.append(f["url"])
        cid = citations.add(
            run_id=run_id, url=f["url"], title=f["title"], snapshot=f["text"]
        )
        cited_ids.append(cid)
        sources.append({"cid": cid, "url": f["url"], "title": f["title"], "text": f["text"]})

    # 3) 证据构造：混合检索选块（BM25+向量→RRF→可选 rerank）；不可用/无命中回退整页截断
    if sources:
        evidence_blocks = _build_evidence(subquestion, sources, settings)
        if not evidence_blocks:
            evidence_blocks = _fallback_evidence(sources, settings)

    if not evidence_blocks:
        return Finding(
            subquestion=subquestion,
            summary="[unverified] 未检索到可用来源",
            citation_ids=[],
            injected=injected_seen,
            urls=source_urls,
        )

    # 4) 摘要（带引用编号）
    result = await deps_llm.chat(
        [
            {"role": "system", "content": RESEARCHER_SYSTEM},
            {"role": "user", "content": f"子问题：{subquestion}\n\n证据材料：\n\n" + "\n\n".join(evidence_blocks)},
        ],
        model=settings.model_for("light"),  # 调研摘要是结构化任务，light 模型够用，省额度给 synthesizer
        node="researcher",
        temperature=0.2,
        max_tokens=4096,  # 推理模型的思考链会吃 token，给足余量
    )
    summary, used = _parse_summary(result.content, cited_ids)
    return Finding(subquestion=subquestion, summary=summary, citation_ids=used,
                   injected=injected_seen, urls=source_urls)


def _parse_summary(text: str, valid_ids: list[int]) -> tuple[str, list[int]]:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(0))
            summary = str(data.get("summary", "")).strip()
            used = [int(i) for i in data.get("used_citations", []) if int(i) in valid_ids]
            if summary:
                return summary, used
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    # 兜底：全文即 summary，从 [cite:N] 提取 used
    used = [int(n) for n in re.findall(r"\[cite:(\d+)\]", text) if int(n) in valid_ids]
    return text.strip(), sorted(set(used))


async def researcher_node(
    state: ResearchState,
    *,
    llm: MeteredLLM,
    settings: Settings,
    search: Tool,
    fetch: Tool,
    citations: CitationStore,
    trace: RunTrace,
) -> dict:
    run_id = state.get("run_id", "unknown")
    # 任务级规划优先；退化到旧子问题列表时，为每个 query 补一个占位任务
    tasks: list[ResearchTask] = list(state.get("tasks", []))
    subs = [t["query"] for t in tasks if t.get("query")] or state.get("subquestions", [])
    if not tasks and subs:
        tasks = [ResearchTask(id=i, title=sub, intent="", query=sub, status="pending")
                 for i, sub in enumerate(subs, 1)]

    # 按任务并行调研：工具调用是 IO 密集，并发能把串行压到 ~1/4 耗时
    # 单个任务失败不影响其他（返回 unverified finding 而非抛异常）
    async def safe_research(task: ResearchTask) -> Finding:
        try:
            return await _research_one(
                task["query"], deps_llm=llm, settings=settings, search=search,
                fetch=fetch, citations=citations, trace=trace, run_id=run_id,
            )
        except Exception as e:  # noqa: BLE001 - 单任务失败降级为 unverified
            return Finding(
                subquestion=task["query"], urls=[],
                summary=f"[unverified] 调研过程出错: {type(e).__name__}",
                citation_ids=[], injected=False,
            )

    findings_new = await asyncio.gather(*[safe_research(task) for task in tasks])

    # 回写任务级状态（展示层），引用链 findings 不变
    # 注意：task 字典已含 status/summary 等字段，需过滤后合并，避免重复关键字
    updated_tasks: list[ResearchTask] = []
    for task, finding in zip(tasks, findings_new):
        status = "completed"
        if not finding["citation_ids"] and not finding.get("urls"):
            status = "skipped" if "调研过程出错" not in finding["summary"] else "failed"
        merged = {k: v for k, v in task.items()
                  if k not in {"status", "summary", "sources", "citation_ids"}}
        merged.update(
            status=status,
            summary=finding["summary"],
            sources=finding.get("urls", []),
            citation_ids=finding["citation_ids"],
        )
        updated_tasks.append(ResearchTask(**merged))

    findings: list[Finding] = list(state.get("findings", [])) + list(findings_new)
    return {"findings": findings, "tasks": updated_tasks}
