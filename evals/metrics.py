"""指标定义（写进 README 的指标表）：

- 任务成功率：人工判定（run_eval 生成待标注清单）
- 引用可验证率：supported / 总引用数       目标 ≥ 85%
- 幻觉结论率：unsupported 结论 / 总结论数   目标 ≤ 10%
- 检索命中率：金标准来源出现在引用中的比例  目标 ≥ 70%
- 单题成本：token 消耗均值
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EvalRow:
    qid: str
    run_id: str = ""
    gold_urls: list[str] = field(default_factory=list)
    cited_urls: list[str] = field(default_factory=list)
    supported: int = 0
    partial: int = 0
    unsupported: int = 0
    total_claims: int = 0
    tokens: int = 0
    latency_s: float = 0.0
    tool_success: int = 0
    tool_total: int = 0
    budget_exhausted: bool = False
    error: str = ""


def aggregate(rows: list[EvalRow]) -> dict[str, float | int]:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    total_cited = sum(r.supported + r.partial + r.unsupported for r in rows)
    total_claims = sum(r.total_claims for r in rows)
    hit_rows = sum(
        1 for r in rows
        if r.gold_urls and any(any(g in c for c in r.cited_urls) for g in r.gold_urls)
    )
    return {
        "n": n,
        "引用可验证率": sum(r.supported for r in rows) / total_cited if total_cited else 0.0,
        "幻觉结论率": sum(r.unsupported for r in rows) / total_claims if total_claims else 0.0,
        "检索命中率": hit_rows / n,
        "单题成本tokens": sum(r.tokens for r in rows) / n,
        "工具成功率": (
            sum(r.tool_success for r in rows) / sum(r.tool_total for r in rows)
            if sum(r.tool_total for r in rows) else 0.0
        ),
        "平均耗时s": sum(r.latency_s for r in rows) / n,
    }
