"""工具层：错误归一化、幂等重试策略、429 熔断器。"""
import asyncio

import pytest

from dra.provider.base import CircuitBreaker
from dra.tools.base import ErrorType, ToolResult, call_tool


class FlakyTool:
    name = "flaky"
    idempotent = True

    def __init__(self, fail_times: int):
        self.fail_times = fail_times
        self.calls = 0

    async def run(self, params):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise ConnectionError("boom")
        return ToolResult(ok=True, data={"n": self.calls})


class NonIdemTool(FlakyTool):
    name = "nonidem"
    idempotent = False


@pytest.mark.asyncio
async def test_idempotent_retries_until_success():
    tool = FlakyTool(fail_times=1)
    res = await call_tool(tool, {}, retries=2, retry_backoff_s=0)
    assert res.ok and res.data["n"] == 2


@pytest.mark.asyncio
async def test_non_idempotent_no_retry():
    tool = NonIdemTool(fail_times=5)
    res = await call_tool(tool, {}, retries=3, retry_backoff_s=0)
    assert not res.ok
    assert res.error_type == ErrorType.NETWORK
    assert tool.calls == 1


@pytest.mark.asyncio
async def test_timeout_normalized():
    class Slow:
        name = "slow"
        idempotent = False

        async def run(self, params):
            await asyncio.sleep(5)
            return ToolResult(ok=True)

    res = await call_tool(Slow(), {}, timeout_s=0.05)
    assert not res.ok
    assert res.error_type == ErrorType.TIMEOUT


def test_circuit_breaker_opens_and_recovers():
    cb = CircuitBreaker(cooldown_s=0.01, fail_threshold=2)
    assert cb.allow()
    cb.on_rate_limited()
    assert cb.allow()  # 第一次未达阈值
    cb.on_rate_limited()
    assert not cb.allow()  # 熔断打开
    import time

    time.sleep(0.02)
    assert cb.allow()  # 冷却后半开
    cb.on_success()
    assert cb.allow()
