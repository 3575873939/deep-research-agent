"""人工介入节点：开关行为 + 路由 + interrupt/resume 往返。"""
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command

from dra.config import Settings
from dra.graph.builder import extract_interrupt
from dra.graph.human import human_review_node, route_after_human, route_after_verify
from dra.graph.state import ResearchState


def _state(verdicts: list[str], round_: int = 0) -> dict:
    return {
        "question": "q",
        "verify_report": [
            {"claim": f"结论{i}", "citation_id": i + 1, "verdict": v}
            for i, v in enumerate(verdicts)
        ],
        "verify_round": round_,
    }


async def test_disabled_never_interrupts():
    s = Settings(dra_human_review=False)
    out = await human_review_node(_state(["unsupported"]), settings=s)
    assert out["human_decision"] == "auto"


async def test_enabled_but_no_risky_returns_auto():
    # 全部 supported → 没有要人看的结论，不打断
    s = Settings(dra_human_review=True)
    out = await human_review_node(_state(["supported", "supported"]), settings=s)
    assert out["human_decision"] == "auto"


def test_route_after_verify_prioritizes_auto_redo():
    # unsupported 超 20% 且未达轮次上限 → 自动打回，不进人工确认
    s = Settings(dra_human_review=True)
    assert route_after_verify(_state(["unsupported", "supported"], round_=0), s) == "redo"
    # 达轮次上限 → 转人工确认
    assert route_after_verify(_state(["unsupported"] * 3, round_=2), s) == "review"
    # 开关关闭 → 直接定稿
    assert route_after_verify(_state(["supported"]), Settings(dra_human_review=False)) == "finalize"


def test_route_after_human_respects_round_limit():
    s = Settings(dra_human_review=True, verifier_max_rounds=2)
    assert route_after_human({"human_decision": "redo", "verify_round": 0}, s) == "redo"
    # 已达轮次上限：即使人选 redo 也不能再回调研，避免死循环
    assert route_after_human({"human_decision": "redo", "verify_round": 2}, s) == "finalize"
    assert route_after_human({"human_decision": "accept"}, s) == "finalize"


async def test_interrupt_then_resume_roundtrip(tmp_path):
    """核心验证：interrupt 真的会暂停，Command(resume=...) 真的能带着人选的决策继续。"""
    settings = Settings(dra_human_review=True, human_review_min_risky=1)

    async def _review(state: dict) -> dict:
        return await human_review_node(state, settings=settings)

    g = StateGraph(ResearchState)
    g.add_node("review", _review)
    g.set_entry_point("review")
    g.add_edge("review", END)

    config = {"configurable": {"thread_id": "t-roundtrip"}}
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "cp.db")) as cp:
        graph = g.compile(checkpointer=cp)

        paused = await graph.ainvoke(_state(["unsupported", "supported"]), config)
        payload = extract_interrupt(paused)
        assert payload is not None, "应当因存在 unsupported 结论而暂停"
        assert payload["value"]["kind"] == "report_approval"
        assert payload["value"]["risky_count"] == 1
        assert [o["value"] for o in payload["value"]["options"]] == ["accept", "redo"]

        # 带着人选的决定恢复：状态里应写入 redo
        resumed = await graph.ainvoke(Command(resume={"action": "redo"}), config)
        assert resumed["human_decision"] == "redo"


async def test_extract_interrupt_returns_none_when_absent():
    assert extract_interrupt({}) is None
    assert extract_interrupt({"draft": "x"}) is None
