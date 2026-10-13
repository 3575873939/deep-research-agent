"""CLI 入口：python -m dra.cli <问题>

若开启人工确认（DRA_HUMAN_REVIEW=true）且 verifier 检出缺乏引用支持的结论，
流程会在报告定稿前暂停，在此交互式询问后从断点恢复。
"""
from __future__ import annotations

import argparse
import asyncio
import sys

from .graph.builder import ResearchRuntime, extract_interrupt


def _ask_human(info: dict) -> dict:
    """把待确认内容打印给人看，返回决策。非交互环境（管道/CI）默认接受。"""
    print(f"\n[人工确认] {info.get('message', '')}")
    for i, c in enumerate(info.get("risky_claims", []), 1):
        print(f"  {i}. [{c.get('verdict')}] [cite:{c.get('citation_id')}] "
              f"{str(c.get('claim', ''))[:120]}")
    if not sys.stdin.isatty():
        print("（非交互环境，默认接受当前报告）")
        return {"action": "accept"}
    ans = input("接受当前报告？(Y=接受 / n=打回重新调研): ").strip().lower()
    return {"action": "accept" if ans in ("", "y", "yes") else "redo"}


async def _list_mcp_tools(rt: ResearchRuntime) -> None:
    """列出 MCP server 暴露的工具（验证协议连通性，也方便挑选检索工具）。"""
    from .tools.mcp_client import open_toolbox, parse_mcp_servers

    configs = parse_mcp_servers(rt.settings.dra_mcp_servers)
    if not configs:
        print("未配置 MCP server（DRA_MCP_SERVERS 为空）。示例：")
        print('  DRA_MCP_SERVERS=[{"name":"fs","command":"npx",'
              '"args":["-y","@modelcontextprotocol/server-filesystem","/tmp"]}]')
        return

    print(f"连接 {len(configs)} 个 MCP server ...")
    async with open_toolbox(configs) as tools:
        if not tools:
            print("未发现任何工具")
            return
        for name, t in sorted(tools.items()):
            mark = "只读" if t.idempotent else "可写"
            print(f"  {name}  [{mark}]  {t.description[:80]}")

    current = rt.settings.dra_mcp_search_tool.strip()
    if current:
        ok = current in tools
        print(f"\n当前检索源: {current}" + ("" if ok else "  （未找到，运行时将降级到内置检索）"))


def main() -> None:
    ap = argparse.ArgumentParser(prog="dra")
    ap.add_argument("question", nargs="?", help="研究问题")
    ap.add_argument("--show-citations", action="store_true", help="报告后附引用快照清单")
    ap.add_argument("--human-review", action="store_true",
                    help="本次开启人工确认（覆盖环境配置）")
    ap.add_argument("--list-mcp-tools", action="store_true",
                    help="列出 DRA_MCP_SERVERS 配置的 MCP 工具后退出")
    args = ap.parse_args()

    rt = ResearchRuntime()

    if args.list_mcp_tools:
        asyncio.run(_list_mcp_tools(rt))
        return

    if not args.question:
        ap.error("需要提供研究问题（或使用 --list-mcp-tools）")

    if args.human_review:
        rt.settings = rt.settings.model_copy(update={"dra_human_review": True})

    async def _run():
        return await rt.run(args.question)

    state, trace = asyncio.run(_run())

    # 人工确认断点：询问后从断点恢复（resume 不触发新的 LLM 调用）
    payload = extract_interrupt(state)
    if payload is not None:
        decision = _ask_human(payload["value"] or {})
        state, trace = asyncio.run(rt.resume(trace.run_id, decision))

    report = state.get("draft", state.get("final_report", "(无输出)"))
    print(report)
    print(f"\n---\nrun_id={trace.run_id} tokens={trace.total_tokens} "
          f"latency={trace.latency_s:.1f}s budget_exhausted={trace.budget_exhausted}")
    if state.get("human_decision"):
        print(f"human_decision={state['human_decision']}")
    verify = state.get("verify_report", [])
    if verify:
        sup = sum(1 for v in verify if v["verdict"] == "supported")
        print(f"verify: supported={sup} partial={sum(1 for v in verify if v['verdict'] == 'partial')} "
              f"unsupported={sum(1 for v in verify if v['verdict'] == 'unsupported')}")
    if args.show_citations:
        for c in rt.citations.list_for_run(trace.run_id):
            print(f"[cite:{c.id}] {c.title} | {c.url} | hash={c.content_hash[:12]}")


if __name__ == "__main__":
    main()
