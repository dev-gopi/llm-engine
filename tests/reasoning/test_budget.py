import pytest

from runtime.reasoning import resolve_reasoning_budget


def test_reasoning_budgets_are_deterministic():
    assert resolve_reasoning_budget("none",max_tokens=100).max_tokens == 0
    assert resolve_reasoning_budget("low",max_tokens=1000).max_tokens == 128
    assert resolve_reasoning_budget("medium",max_tokens=1000).max_tokens == 512
    assert resolve_reasoning_budget("high",max_tokens=1000).max_tokens == 1000

def test_invalid_effort_rejected():
    with pytest.raises(ValueError): resolve_reasoning_budget("x",max_tokens=10)
