"""CLI 入口：python -m dra.cli <问题>"""
from __future__ import annotations

import argparse

from .graph.builder import ResearchRuntime
from .tools.citations import CitationStore


def main() -> None:
    ap = argparse.ArgumentParser(prog="dra")
    ap.add_argument("question")
    ap.add_argument("--show-citations", action="store_true", help="报告后附引用快照清单")
    args = ap.parse_args()

    import asyncio

    rt = ResearchRuntime()

    async def _run():
        return await rt.run(args.question)

    state, trace = asyncio.run(_run())
    report = state.get("draft", state.get("final_report", "(无输出)"))
    print(report)
    print(f"\n---\nrun_id={trace.run_id} tokens={trace.total_tokens} "
          f"latency={trace.latency_s:.1f}s budget_exhausted={trace.budget_exhausted}")
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
