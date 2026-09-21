"""Runtime reasoning-effort policy and measurable budget accounting."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

ReasoningEffort = Literal["none", "low", "medium", "high"]

@dataclass(frozen=True)
class ReasoningBudget:
    effort: ReasoningEffort
    max_tokens: int
    min_tokens: int = 0

DEFAULT_BUDGETS = {"none": 0, "low": 128, "medium": 512, "high": 1024}

def resolve_reasoning_budget(effort: str | None, *, max_tokens: int) -> ReasoningBudget:
    value = (effort or "none").lower()
    if value not in DEFAULT_BUDGETS:
        raise ValueError("reasoning_effort must be one of none, low, medium, high")
    if max_tokens < 0:
        raise ValueError("max_tokens must be non-negative")
    return ReasoningBudget(value, min(DEFAULT_BUDGETS[value], max_tokens), 0)
