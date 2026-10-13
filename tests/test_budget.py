"""预算熔断：预检先于调用、超限即抛。"""
import pytest

from dra.guard.budget import BudgetExceeded, BudgetGuard


def test_preflight_pass_then_block():
    g = BudgetGuard(max_tokens=100)
    g.check()  # 不抛
    g.record(30, 40)
    assert g.remaining == 30
    g.record(20, 10)
    assert g.exhausted
    with pytest.raises(BudgetExceeded):
        g.check()


def test_zero_budget_blocks_immediately():
    g = BudgetGuard(max_tokens=0)
    with pytest.raises(BudgetExceeded):
        g.check()
