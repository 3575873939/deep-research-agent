"""MCP 工具层：连接外部 MCP server，把其工具适配为本地 Tool 协议。

设计要点（面试讲点）：
1. **传输走 stdio**——MCP 最常见的本地集成方式，server 作为子进程启动，
   JSON-RPC over stdin/stdout。
2. **适配后与内置工具同构**：同一个 `Tool` 协议 + `ToolResult` 错误归一化，
   调用方无需区分工具来自 MCP server 还是本地实现。
3. **连不上就降级**：MCP 是增强能力，任何 server 启动失败都不能阻断内置
   search/fetch 主链路。
4. **重试纪律沿用 MCP 自身的语义**：只有 `readOnlyHint=true` 的工具才允许重试，
   写操作一律不重试——避免重复副作用。
"""
from __future__ import annotations

import json
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from .base import ErrorType, Tool, ToolResult


@dataclass
class MCPServerConfig:
    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] | None = None

    def to_params(self):
        from mcp import StdioServerParameters

        return StdioServerParameters(command=self.command, args=self.args, env=self.env)


def parse_mcp_servers(raw: str | None) -> list[MCPServerConfig]:
    """解析 DRA_MCP_SERVERS（JSON 数组）。任何异常都返回空列表——配置错误不该拖垮启动。

    期望格式：
        [{"name": "fs", "command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]}]
    """
    if not raw or not raw.strip():
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    out: list[MCPServerConfig] = []
    for item in data if isinstance(data, list) else []:
        if not isinstance(item, dict) or not item.get("name") or not item.get("command"):
            continue
        out.append(MCPServerConfig(
            name=str(item["name"]),
            command=str(item["command"]),
            args=[str(a) for a in item.get("args", [])],
            env=item.get("env"),
        ))
    return out


class MCPTool:
    """把一个 MCP 工具暴露为本地 Tool。name 形如 `<server>.<tool>`，避免跨 server 撞名。"""

    def __init__(self, session: Any, server: str, tool: Any) -> None:
        self._session = session
        self.server = server
        self.remote_name = tool.name
        self.name = f"{server}.{tool.name}"
        self.description = getattr(tool, "description", "") or ""
        self.input_schema = getattr(tool, "inputSchema", None) or {}
        # 只有标注为只读的工具才可重试；未标注的一律按有副作用处理
        annotations = getattr(tool, "annotations", None)
        self.idempotent = bool(getattr(annotations, "readOnlyHint", False))

    async def run(self, params: dict[str, Any]) -> ToolResult:
        try:
            res = await self._session.call_tool(self.remote_name, params)
        except Exception as e:  # noqa: BLE001 - 归一化后交给 call_tool 的重试纪律
            return ToolResult(ok=False, error=str(e)[:300], error_type=ErrorType.UNKNOWN)

        # MCP 返回 content 块列表，本项目只消费文本块
        text = "\n".join(
            c.text for c in (getattr(res, "content", None) or [])
            if getattr(c, "type", "") == "text"
        )
        if getattr(res, "isError", False):
            return ToolResult(ok=False, data=text, error=text[:300],
                              error_type=ErrorType.UNKNOWN)
        return ToolResult(ok=True, data=text)


@asynccontextmanager
async def open_toolbox(configs: list[MCPServerConfig]) -> AsyncIterator[dict[str, Tool]]:
    """连接所有配置的 server，产出 {tool_name: Tool}。退出时统一关闭子进程。

    单个 server 连接失败只跳过它（打印一行日志），其余照常可用。
    """
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client

    tools: dict[str, Tool] = {}
    async with AsyncExitStack() as stack:
        for cfg in configs:
            try:
                read, write = await stack.enter_async_context(stdio_client(cfg.to_params()))
                session = await stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                listed = await session.list_tools()
                for t in listed.tools:
                    mcp_tool = MCPTool(session, cfg.name, t)
                    tools[mcp_tool.name] = mcp_tool
            except Exception as e:  # noqa: BLE001 - 单个 server 失败不影响其它
                print(f"[mcp] server '{cfg.name}' 连接失败，已跳过: {type(e).__name__}: {e}")
        yield tools
