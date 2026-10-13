"""批量跑评测 + 断点续跑。

用法：
    python evals/run_eval.py [--limit N] [--out evals/results.jsonl]

断点续跑：结果 JSONL 已存在的 qid 跳过——429 停手后次日接着跑，不重烧 token。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dra.graph.builder import ResearchRuntime  # noqa: E402
from dra.provider.base import RateLimited  # noqa: E402

from metrics import EvalRow, aggregate  # noqa: E402

DATASET = Path(__file__).parent / "dataset.jsonl"


def load_dataset(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_done(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    with out_path.open(encoding="utf-8") as f:
        return {json.loads(line)["qid"] for line in f if line.strip()}


async def eval_one(rt: ResearchRuntime, item: dict) -> EvalRow:
    row = EvalRow(qid=item["qid"], gold_urls=item.get("gold_urls", []))
    try:
        state, trace = await rt.run(item["question"])
    except RateLimited:
        row.error = "rate_limited_circuit_open"
        raise
    except Exception as e:  # noqa: BLE001 - 单题失败不中断整批
        row.error = str(e)[:300]
        return row

    row.run_id = trace.run_id
    row.tokens = trace.total_tokens
    row.latency_s = trace.latency_s
    row.budget_exhausted = trace.budget_exhausted
    row.tool_total = len(trace.tool_calls)
    row.tool_success = sum(1 for t in trace.tool_calls if t["ok"])

    verify = state.get("verify_report", [])
    row.supported = sum(1 for v in verify if v["verdict"] == "supported")
    row.partial = sum(1 for v in verify if v["verdict"] == "partial")
    row.unsupported = sum(1 for v in verify if v["verdict"] == "unsupported")
    row.total_claims = len(verify)

    cites = rt.citations.list_for_run(trace.run_id)
    row.cited_urls = [c.url for c in cites]

    # 报告落盘，供人工判定任务成功率
    report_path = Path(__file__).parent / "reports" / f"{item['qid']}.md"
    report_path.parent.mkdir(exist_ok=True)
    report_path.write_text(state.get("draft", ""), encoding="utf-8")
    return row


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 题")
    ap.add_argument("--qids", default="", help="只跑指定 qid，逗号分隔，如 t01,p01,s01")
    ap.add_argument("--out", default=str(Path(__file__).parent / "results.jsonl"))
    args = ap.parse_args()

    dataset = load_dataset(DATASET)
    if args.qids:
        wanted = set(args.qids.split(","))
        dataset = [d for d in dataset if d["qid"] in wanted]
    if args.limit:
        dataset = dataset[: args.limit]
    out_path = Path(args.out)
    done = load_done(out_path)
    todo = [d for d in dataset if d["qid"] not in done]
    print(f"dataset={len(dataset)} done={len(done)} todo={len(todo)}")

    rt = ResearchRuntime()
    rows: list[EvalRow] = []
    with out_path.open("a", encoding="utf-8") as f:
        for item in todo:
            print(f"[{item['qid']}] {item['question'][:50]}...")
            try:
                row = await eval_one(rt, item)
            except RateLimited:
                print("!! 429 熔断触发，停手。已跑结果已落盘，冷却后直接重跑本脚本即可续跑。")
                break
            rows.append(row)
            f.write(json.dumps(row.__dict__, ensure_ascii=False) + "\n")
            f.flush()
            print(f"    tokens={row.tokens} supported={row.supported} "
                  f"partial={row.partial} unsupported={row.unsupported} err={row.error or '-'}")

    # 汇总含历史已跑结果（断点续跑后的全量指标）
    all_rows = [EvalRow(**json.loads(line)) for line in out_path.open(encoding="utf-8") if line.strip()]
    print("\n== 汇总指标 ==")
    for k, v in aggregate(all_rows).items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    asyncio.run(main())
