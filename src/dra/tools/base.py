"""工具层基座：统一 ToolResult + 超时 + 幂等重试 + 错误归一化。

所有工具（无论是否经 MCP server）都实现 Tool 协议，返回 ToolResult——
错误归一化是"工具层可靠性设计"的面试讲点：eval 里可单独统计各工具成功率。
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol


class ErrorType(str, Enum):
    NONE = "none"
    TIMEOUT = "timeout"
    NETWORK = "network"
    RATE_LIMITED = "rate_limited"
    AUTH = "auth"
    BAD_INPUT = "bad_input"
    EMPTY = "empty"
    UNKNOWN = "unknown"


@dataclass
class ToolResult:
    ok: bool
    data: Any = None
    error: str = ""
    error_type: ErrorType = ErrorType.NONE
    latency_s: float = 0.0
    raw_len: int = 0
    tool: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def to_context(self) -> str:
        """给 LLM 看的文本形式。"""
        if not self.ok:
            return f"[tool:{self.tool}] ERROR({self.error_type.value}): {self.error}"
        return str(self.data)


class Tool(Protocol):
    name: str
    # 幂等操作才允许重试：search 重试、browser 不重试
    idempotent: bool

    async def run(self, params: dict[str, Any]) -> ToolResult: ...


def classify_error(exc: BaseException) -> ErrorType:
    msg = str(exc).lower()
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return ErrorType.TIMEOUT
    if "429" in msg or "rate limit" in msg:
        return ErrorType.RATE_LIMITED
    if "401" in msg or "403" in msg or "api key" in msg or "unauthorized" in msg:
        return ErrorType.AUTH
    if isinstance(exc, (ConnectionError, OSError)):
        return ErrorType.NETWORK
    return ErrorType.UNKNOWN


async def call_tool(
    tool: Tool,
    params: dict[str, Any],
    *,
    timeout_s: float = 30.0,
    retries: int = 2,
    retry_backoff_s: float = 1.0,
) -> ToolResult:
    """统一调用入口：超时控制 + 幂等重试 + 错误归一化。"""
    attempts = 1 + (retries if tool.idempotent else 0)
    last: ToolResult | None = None
    for attempt in range(attempts):
        started = time.monotonic()
        try:
            result = await asyncio.wait_for(tool.run(params), timeout=timeout_s)
            result.tool = tool.name
            result.latency_s = time.monotonic() - started
            result.raw_len = len(str(result.data)) if result.data is not None else 0
            if result.ok or not tool.idempotent:
                return result
            last = result
        except Exception as e:  # noqa: BLE001 - 错误归一化正是这里的目的
            last = ToolResult(
                ok=False,
                tool=tool.name,
                error=str(e)[:500],
                error_type=classify_error(e),
                latency_s=time.monotonic() - started,
            )
        if attempt < attempts - 1:
            await asyncio.sleep(retry_backoff_s * (attempt + 1))
    assert last is not None
    return last
