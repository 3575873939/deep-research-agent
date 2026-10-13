"""引用库 + verifier 规则函数。"""
import re

from dra.graph.verifier import extract_claims, _parse_verdict
from dra.tools.citations import CitationStore


def test_citation_store_roundtrip(tmp_path):
    store = CitationStore(str(tmp_path / "t.db"))
    cid = store.add(run_id="r1", url="http://a", title="T", snapshot="原文快照内容")
    cite = store.get(cid)
    assert cite is not None
    assert cite.url == "http://a"
    assert len(cite.content_hash) == 64
    assert store.list_for_run("r1")[0].id == cid
    store.close()


def test_extract_claims_splits_and_finds_cites():
    draft = "## 核心结论\nPython 3.13 移除了 GIL [cite:3]。这是一个重大变化。\nLangGraph 支持 checkpoint [cite:7]。"
    claims = extract_claims(draft)
    cited = [c for c in claims if c[1] is not None]
    uncited = [c for c in claims if c[1] is None]
    assert any(cid == 3 for _, cid in cited)
    assert any(cid == 7 for _, cid in cited)
    assert uncited  # "这是一个重大变化。" 无引用


def test_parse_verdict_strict():
    assert _parse_verdict("supported\n理由：原文明确提及") == "supported"
    assert _parse_verdict("PARTIAL") == "partial"
    assert _parse_verdict("我不确定") == "unsupported"  # 异常输出从严
