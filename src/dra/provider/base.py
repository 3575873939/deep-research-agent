"""LLM 通道抽象：统一计量接口 + 预算预检 + 429 熔断。

设计要点（面试讲点）：
1. 每次调用前 BudgetGuard.check()，超限抛 BudgetExceeded → graph 降级
2. 429 熔断器：连续限流触发冷却期，期间直接 fail-fast，不再打 API
3. 节点级路由：get_llm("researcher") 拿到该节点配置的模型
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..guard.budget import BudgetExceeded, BudgetGuard


class RateLimited(Exception):
    """上游 429。熔断器据此累计。"""


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class ChatResult:
    content: str
    usage: Usage
    model: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


class LLMProvider(Protocol):
    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> ChatResult: ...


class CircuitBreaker:
    """429 熔断：冷却期内直接拒绝，避免无效重试烧钱（复用 429 停手纪律的工程化）。"""

    def __init__(self, cooldown_s: float, fail_threshold: int = 2) -> None:
        self.cooldown_s = cooldown_s
        self.fail_threshold = fail_threshold
        self._consecutive_failures = 0
        self._opened_at: float | None = None

    def allow(self) -> bool:
        if self._opened_at is None:
            return True
        if time.monotonic() - self._opened_at >= self.cooldown_s:
            # 冷却结束，半开试探
            self._opened_at = None
            self._consecutive_failures = 0
            return True
        return False

    def on_success(self) -> None:
        self._consecutive_failures = 0
        self._opened_at = None

    def on_rate_limited(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.fail_threshold:
            self._opened_at = time.monotonic()

    @property
    def is_open(self) -> bool:
        return not self.allow()


class MeteredLLM:
    """在 provider 外面包一层：预算预检 + 计量 + 熔断。所有节点调用都走这里。"""

    def __init__(
        self,
        provider: LLMProvider,
        budget: BudgetGuard,
        breaker: CircuitBreaker,
        on_usage: Any = None,  # Callable[[node, Usage], None] 接 trace
        min_interval_s: float = 0.0,
    ) -> None:
        self.provider = provider
        self.budget = budget
        self.breaker = breaker
        self.on_usage = on_usage
        self.min_interval_s = min_interval_s
        self._last_call_ts = 0.0
        self._pace_lock = asyncio.Lock()

    async def _pace(self) -> None:
        """客户端节流：保证两次调用间隔 ≥ min_interval_s。

        免费档（如 AIHubMix 5 请求/分钟）下，硬打会连续 429 → 触发熔断，额度全浪费在
        失败重试上。锁保证并发节点（如 researcher 的并行子任务）也按同一节奏排队。
        """
        if self.min_interval_s <= 0:
            return
        async with self._pace_lock:
            wait = self.min_interval_s - (time.monotonic() - self._last_call_ts)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_call_ts = time.monotonic()

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str,
        node: str = "unknown",
        **kwargs: Any,
    ) -> ChatResult:
        # 预算预检：超限直接抛，不发请求
        self.budget.check()
        if not self.breaker.allow():
            raise RateLimited(f"circuit open, cooldown {self.breaker.cooldown_s}s")

        await self._pace()

        try:
            result = await self.provider.chat(messages, model=model, **kwargs)
        except RateLimited:
            self.breaker.on_rate_limited()
            raise

        self.breaker.on_success()
        self.budget.record(result.usage.prompt_tokens, result.usage.completion_tokens)
        if self.on_usage:
            self.on_usage(node, result.usage)
        return result


__all__ = [
    "BudgetExceeded",
    "ChatResult",
    "CircuitBreaker",
    "LLMProvider",
    "MeteredLLM",
    "RateLimited",
    "Usage",
]
