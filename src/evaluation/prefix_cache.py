"""Prefix-cache observability and accounting."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PrefixCacheMetrics:
    hits: int
    misses: int
    cached_tokens: int
    prefill_tokens_saved: int
    memory_bytes: int
    evictions: int

    def to_dict(self) -> dict[str, int]:
        return self.__dict__.copy()


def collect_prefix_cache_metrics(generator: Any) -> PrefixCacheMetrics:
    cache = getattr(generator, "prefix_cache", None)
    return PrefixCacheMetrics(
        hits=int(getattr(generator, "prefix_cache_hits", 0)),
        misses=int(getattr(generator, "prefix_cache_misses", 0)),
        cached_tokens=int(getattr(generator, "prefix_cache_tokens", 0)),
        prefill_tokens_saved=int(getattr(generator, "prefix_prefill_tokens_saved", 0)),
        memory_bytes=int(getattr(cache, "memory_bytes", 0) if cache is not None else 0),
        evictions=int(getattr(cache, "evictions", 0) if cache is not None else 0),
    )
