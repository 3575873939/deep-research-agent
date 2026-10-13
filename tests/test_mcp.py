"""MCP 工具层：配置解析 / 适配语义 / 真实 stdio 往返。"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from dra.tools.base import call_tool
from dra.tools.mcp_client import (
    MCPServerConfig,
    MCPTool,
    open_toolbox,
    parse_mcp_servers,
)

ECHO_SERVER = Path(__file__).parent / "fixtures" / "mcp_echo_server.py"


# ---------- 配置解析 ----------

def test_parse_valid_config():
    cfg = parse_mcp_servers(
        '[{"name":"fs","command":"npx","args":["-y","pkg","/tmp"]}]')
    assert len(cfg) == 1
    assert cfg[0].name == "fs"
    assert cfg[0].command == "npx"
    assert cfg[0].args == ["-y", "pkg", "/tmp"]


@pytest.mark.parametrize("raw", ["", None, "not json", "{}", '[{"name":"x"}]', "[]"])
def test_parse_invalid_returns_empty(raw):
    # 配置错误一律降级为空列表，不该拖垮启动
    assert parse_mcp_servers(raw) == []


def test_to_params_carries_fields():
    cfg = MCPServerConfig(name="a", command="python", args=["-m", "x"], env={"K": "V"})
    p = cfg.to_params()
    assert p.command == "python" and p.args == ["-m", "x"] and p.env == {"K": "V"}


# ---------- 适配语义 ----------

def _fake_tool(name="t", description="d", annotations=None):
    return SimpleNamespace(name=name, description=description,
                           inputSchema={"type": "object"}, annotations=annotations)


def _fake_session(result):
    class _S:
        def __init__(self):
            self.calls = []

        async def call_tool(self, name, args):
            self.calls.append((name, args))
            return result
    return _S()


def test_readonly_hint_controls_retry():
    ro = MCPTool(_fake_session(None), "srv", _fake_tool(annotations=SimpleNamespace(readOnlyHint=True)))
    rw = MCPTool(_fake_session(None), "srv", _fake_tool())
    assert ro.idempotent is True      # 标注只读 → 允许重试
    assert rw.idempotent is False     # 未标注 → 按有副作用处理，不重试
    assert rw.name == "srv.t" and rw.remote_name == "t"


async def test_run_joins_text_blocks():
    content = [SimpleNamespace(type="text", text="a"),
               SimpleNamespace(type="image", data="x"),
               SimpleNamespace(type="text", text="b")]
    session = _fake_session(SimpleNamespace(content=content, isError=False))
    tool = MCPTool(session, "srv", _fake_tool())
    res = await tool.run({"q": 1})
    assert res.ok and res.data == "a\nb"     # 只消费文本块
    assert session.calls == [("t", {"q": 1})]


async def test_run_maps_is_error_to_failed_result():
    session = _fake_session(SimpleNamespace(
        content=[SimpleNamespace(type="text", text="boom")], isError=True))
    res = await MCPTool(session, "srv", _fake_tool()).run({})
    assert res.ok is False and "boom" in res.error


async def test_run_normalizes_exception():
    class _Boom:
        async def call_tool(self, name, args):
            raise RuntimeError("连接断了")

    res = await MCPTool(_Boom(), "srv", _fake_tool()).run({})
    assert res.ok is False and "连接断了" in res.error


# ---------- 真实 stdio 往返 ----------

async def test_real_stdio_roundtrip_and_call_tool():
    """启动真实 MCP server 子进程，验证连接/列工具/调用/统一入口全链路。"""
    cfg = MCPServerConfig(name="echo", command=sys.executable, args=[str(ECHO_SERVER)])
    async with open_toolbox([cfg]) as tools:
        assert set(tools) == {"echo.echo", "echo.word_count"}, tools

        echoed = await call_tool(tools["echo.echo"], {"text": "hello mcp"}, timeout_s=30)
        assert echoed.ok, echoed.error
        assert "hello mcp" in str(echoed.data)

        counted = await call_tool(tools["echo.word_count"], {"text": "a b c d"}, timeout_s=30)
        assert counted.ok and str(counted.data).strip() == "4"
        # call_tool 会回填工具名与耗时（错误归一化链路的既有行为）
        assert counted.tool == "echo.word_count" and counted.latency_s >= 0


async def test_bad_server_is_skipped_without_breaking_others():
    bad = MCPServerConfig(name="bad", command="definitely-not-a-real-binary-xyz")
    good = MCPServerConfig(name="echo", command=sys.executable, args=[str(ECHO_SERVER)])
    async with open_toolbox([bad, good]) as tools:
        # bad 被跳过，good 仍可用
        assert "echo.echo" in tools
        assert not any(k.startswith("bad.") for k in tools)
