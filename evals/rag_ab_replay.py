# -*- coding: utf-8 -*-
"""零 LLM 成本的确定性对照：用已存引用快照，把同一批材料分别过 RAG 选块 / 整页截断。

分组依据：一个任务内 5 次 citations.add 之间没有 await，所以同任务的引用 id 必然连续。
子问题与分组的配对用词面重合度，配对不精确只会**低估** RAG 的压缩效果（选到相关性更低的块）。
"""
import sys, io, json, os, sqlite3, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, r"d:\Deep Research Agent\deep-research-agent\src")

from dra.config import Settings
from dra.graph.researcher import _build_evidence, _fallback_evidence

BASE = r"d:\Deep Research Agent\deep-research-agent"
GROUP = 5

row = json.loads(open(os.path.join(BASE, "evals", "results_ms_rag_on.jsonl"), encoding="utf-8").readline())
run_id = row["run_id"]
trace = next(json.loads(l) for l in open(os.path.join(BASE, "data", "traces.jsonl"), encoding="utf-8")
             if l.strip() and json.loads(l).get("run_id") == run_id)

conn = sqlite3.connect(os.path.join(BASE, "data", "research.db"))
rows = conn.execute(
    "SELECT id, url, title, snapshot FROM citations WHERE run_id=? ORDER BY id", (run_id,)
).fetchall()
groups = [rows[i:i + GROUP] for i in range(0, len(rows), GROUP)]
subs = trace["subquestions"]
print(f"run={run_id}  引用 {len(rows)} 条 → {len(groups)} 组 × {GROUP}；子问题 {len(subs)} 个\n")

def toks(t):
    return set(re.findall(r"[a-z0-9]+", t.lower())) | set(re.findall(r"[\u4e00-\u9fff]", t))

# 配对：每个子问题挑词面重合度最高的未用分组
def overlap(q, g):
    qt = toks(q)
    return sum(len(qt & toks(r[1] + " " + r[2])) for r in g)

pairs, used = [], set()
for q in subs:
    best = max((i for i in range(len(groups)) if i not in used),
               key=lambda i: overlap(q, groups[i]), default=None)
    if best is not None:
        used.add(best)
        pairs.append((q, groups[best]))

settings = Settings(rag_enabled=True, rag_use_vector=True, rag_use_reranker=False, rag_top_k=16)
settings_off = Settings(rag_enabled=False, rag_use_vector=False, rag_use_reranker=False)

print(f"{'子问题':<44} {'来源':>4} {'选块字符':>9} {'整页字符':>9} {'命中来源':>8}")
print("-" * 82)
tot_on = tot_off = 0
for q, g in pairs:
    sources = [{"cid": r[0], "url": r[1], "title": r[2], "text": r[3]} for r in g]
    on = _build_evidence(q, sources, settings)
    off = _fallback_evidence(sources, settings_off)
    con, coff = sum(map(len, on)), sum(map(len, off))
    tot_on += con; tot_off += coff
    print(f"{q[:42]:<44} {len(g):>4} {con:>9} {coff:>9} {len(on):>8}")

print("-" * 82)
print(f"{'合计':<44} {'':>4} {tot_on:>9} {tot_off:>9}")
print(f"\ncontext 体积：{tot_off} → {tot_on} 字符，压缩 {1 - tot_on / tot_off:.1%}")

# 用本次真实 researcher token 数反推字符/token 比，再换算省下的 token
res_tokens = trace["token_usage"]["researcher"]
print(f"\n本次 researcher 实际 token = {res_tokens}（输入含上下文字符 ≈ {tot_on} + 提示词）")
print(f"→ 该内容字符/token 比 ≈ {tot_on / res_tokens:.2f}")
print(f"→ 省下的字符 ≈ {tot_off - tot_on}，折合 ≈ {(tot_off - tot_on) / (tot_on / res_tokens):.0f} tokens/题")