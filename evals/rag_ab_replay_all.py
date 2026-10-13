# -*- coding: utf-8 -*-
"""把 RAG 确定性对照扩展到历史所有 run：用已存快照重放两条路径，给出压缩率分布。

前提（已在本项目验证）：一个任务内 5 次 citations.add 之间没有 await，
所以同任务的引用 id 连续 → 可按连续的 5 条一组切分出「任务」。
"""
import sys, io, json, os, sqlite3, re, statistics
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, r"d:\Deep Research Agent\deep-research-agent\src")

from dra.config import Settings
from dra.graph.researcher import _build_evidence, _fallback_evidence

BASE = r"d:\Deep Research Agent\deep-research-agent"
GROUP = 5

traces = {}
for line in open(os.path.join(BASE, "data", "traces.jsonl"), encoding="utf-8"):
    if line.strip():
        d = json.loads(line)
        if d.get("subquestions"):
            traces[d["run_id"]] = d

conn = sqlite3.connect(os.path.join(BASE, "data", "research.db"))
settings_on = Settings(rag_enabled=True, rag_use_vector=True, rag_use_reranker=False, rag_top_k=16)
settings_off = Settings(rag_enabled=False, rag_use_vector=False, rag_use_reranker=False)

def toks(t):
    return set(re.findall(r"[a-z0-9]+", t.lower())) | set(re.findall(r"[\u4e00-\u9fff]", t))

def overlap(q, g):
    qt = toks(q)
    return sum(len(qt & toks(r[1] + " " + r[2])) for r in g)

results, skipped = [], 0
for i, (run_id, tr) in enumerate(traces.items(), 1):
    rows = conn.execute(
        "SELECT id, url, title, snapshot FROM citations WHERE run_id=? ORDER BY id", (run_id,)
    ).fetchall()
    if len(rows) < 10 or len(rows) % GROUP != 0:
        skipped += 1
        continue
    ids = [r[0] for r in rows]
    if ids != list(range(ids[0], ids[0] + len(ids))):   # 必须连续，否则分组不可靠
        skipped += 1
        continue
    groups = [rows[k:k + GROUP] for k in range(0, len(rows), GROUP)]
    subs = tr["subquestions"]
    used, pairs = set(), []
    for q in subs:
        avail = [j for j in range(len(groups)) if j not in used]
        if not avail:
            break
        j = max(avail, key=lambda j: overlap(q, groups[j]))
        used.add(j)
        pairs.append((q, groups[j]))
    if len(pairs) < 2:
        skipped += 1
        continue

    con = coff = 0
    for q, g in pairs:
        src = [{"cid": r[0], "url": r[1], "title": r[2], "text": r[3]} for r in g]
        con += sum(map(len, _build_evidence(q, src, settings_on)))
        coff += sum(map(len, _fallback_evidence(src, settings_off)))
    if coff:
        results.append({"run_id": run_id, "tasks": len(pairs), "on": con, "off": coff,
                        "ratio": 1 - con / coff, "question": tr.get("question", "")[:50]})

print(f"扫描 run={len(traces)}  有效={len(results)}  跳过={skipped}\n")
ratios = [r["ratio"] for r in results]
if ratios:
    print(f"压缩率：均值 {statistics.mean(ratios):.1%}  中位 {statistics.median(ratios):.1%}  "
          f"范围 {min(ratios):.1%} ~ {max(ratios):.1%}  (n={len(ratios)})")
    tot_on = sum(r["on"] for r in results); tot_off = sum(r["off"] for r in results)
    print(f"总字符：{tot_off} → {tot_on}   综合压缩 {1 - tot_on / tot_off:.1%}")
    print("\n压缩率最低的 5 个 run（RAG 效果最弱）：")
    for r in sorted(results, key=lambda x: x["ratio"])[:5]:
        print(f"  {r['ratio']:6.1%}  {r['off']:>7}->{r['on']:<7} tasks={r['tasks']}  {r['question']}")