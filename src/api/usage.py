"""Standardized token/cache usage accounting."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UsageAccounting:
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    prefix_cache_hit: bool = False
    prefix_cache_miss: bool = False
    @property
    def total_tokens(self) -> int: return self.prompt_tokens + self.completion_tokens
    def as_dict(self) -> dict[str, int | dict]:
        return {"prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens, "total_tokens": self.total_tokens,
                "cached_tokens": self.cached_tokens, "reasoning_tokens": self.reasoning_tokens,
                "cache": {"prefix_cache_hit": self.prefix_cache_hit, "prefix_cache_miss": self.prefix_cache_miss}}

def account_usage(prompt_tokens: int, completion_tokens: int, *, cached_tokens: int = 0, reasoning_tokens: int = 0, prefix_cache_hit: bool = False, prefix_cache_miss: bool = False) -> UsageAccounting:
    values = (prompt_tokens, completion_tokens, cached_tokens, reasoning_tokens)
    if any(not isinstance(v, int) or v < 0 for v in values): raise ValueError("usage counts must be non-negative integers")
    if cached_tokens > prompt_tokens: raise ValueError("cached_tokens cannot exceed prompt_tokens")
    if reasoning_tokens > completion_tokens: raise ValueError("reasoning_tokens cannot exceed completion_tokens")
    return UsageAccounting(*values, prefix_cache_hit=prefix_cache_hit, prefix_cache_miss=prefix_cache_miss)
