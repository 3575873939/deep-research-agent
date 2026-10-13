"""预算守卫：熔断前置——不是事后统计，而是每次 LLM 调用前检查余量。"""
from __future__ import annotations


class BudgetExceeded(Exception):
    """运行级预算耗尽，graph 捕获后降级：停止调研，用已有材料成稿并标注。"""


class BudgetGuard:
    """单次运行的 token 预算账本。所有节点共享一个实例。"""

    def __init__(self, max_tokens: int) -> None:
        self.max_tokens = max_tokens
        self.used = 0

    def check(self) -> None:
        """调用前预检：已超限则直接抛，不浪费一次 API 调用。"""
        if self.used >= self.max_tokens:
            raise BudgetExceeded(f"budget exhausted: {self.used}/{self.max_tokens} tokens")

    def record(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.used += prompt_tokens + completion_tokens

    @property
    def remaining(self) -> int:
        return max(0, self.max_tokens - self.used)

    @property
    def exhausted(self) -> bool:
        return self.used >= self.max_tokens
