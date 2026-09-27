"""Optional Redis-backed distributed semantic-cache primitives.

Redis stores metadata/vectors as JSON so the implementation works with a
plain Redis deployment.  The backend is intentionally conservative: vector
similarity is calculated client-side over namespace/policy candidates. A
future Redis Vector Search adapter can replace candidate discovery without
changing the public contract.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable

from .semantic_cache_policy import CacheQuota

try:
    import redis.asyncio as redis
except ImportError:  # pragma: no cover - optional dependency
    redis = None


@dataclass(frozen=True)
class RedisCacheConfig:
    url: str = "redis://localhost:6379/0"
    key_prefix: str = "llm-engine:semantic-cache"
    lock_ttl_ms: int = 30_000
    quota: CacheQuota = CacheQuota()

    def __post_init__(self) -> None:
        if self.lock_ttl_ms < 100:
            raise ValueError("lock_ttl_ms must be at least 100ms")


class RedisSemanticCacheBackend:
    """Async Redis backend with fail-open distributed single-flight."""

    def __init__(self, config: RedisCacheConfig, *, client: Any | None = None) -> None:
        self.config = config
        self.client = client

    async def connect(self) -> None:
        if self.client is not None:
            return
        if redis is None:
            raise RuntimeError("redis package is not installed; install the queue extra")
        self.client = redis.from_url(self.config.url, decode_responses=True)
        await self.client.ping()

    def _key(self, namespace: str, exact_key: str) -> str:
        return f"{self.config.key_prefix}:entry:{namespace}:{exact_key}"

    def _lock_key(self, namespace: str, exact_key: str) -> str:
        return f"{self.config.key_prefix}:lock:{namespace}:{exact_key}"

    def _quota_key(self, namespace: str) -> str:
        return f"{self.config.key_prefix}:quota:{namespace}"

    async def get(self, namespace: str, exact_key: str) -> dict[str, Any] | None:
        if self.client is None:
            return None
        raw = await self.client.get(self._key(namespace, exact_key))
        return json.loads(raw) if raw else None

    async def put(self, namespace: str, exact_key: str, payload: dict[str, Any], ttl_seconds: float) -> bool:
        if self.client is None:
            return False
        key = self._key(namespace, exact_key)
        await self.client.set(key, json.dumps(payload, separators=(",", ":")), ex=max(1, int(ttl_seconds)))
        return True

    async def delete(self, namespace: str, exact_key: str) -> bool:
        if self.client is None:
            return False
        return bool(await self.client.delete(self._key(namespace, exact_key)))

    async def acquire_singleflight(self, namespace: str, exact_key: str, token: str) -> bool:
        if self.client is None:
            return True
        return bool(await self.client.set(self._lock_key(namespace, exact_key), token, nx=True, px=self.config.lock_ttl_ms))

    async def release_singleflight(self, namespace: str, exact_key: str, token: str) -> bool:
        if self.client is None:
            return True
        # Lua compare-and-delete prevents one worker from deleting another worker's lock.
        script = "local v=redis.call('get',KEYS[1]); if v==ARGV[1] then return redis.call('del',KEYS[1]) else return 0 end"
        return bool(await self.client.eval(script, 1, self._lock_key(namespace, exact_key), token))

    async def warm(self, namespace: str, entries: list[tuple[str, dict[str, Any], float]]) -> int:
        count = 0
        for exact_key, payload, ttl in entries:
            count += int(await self.put(namespace, exact_key, payload, ttl))
        return count

    async def purge(self, namespace: str, *, exact_key: str | None = None) -> int:
        if self.client is None:
            return 0
        if exact_key is not None:
            return int(await self.delete(namespace, exact_key))
        pattern = f"{self.config.key_prefix}:entry:{namespace}:*"
        keys = [key async for key in self.client.scan_iter(match=pattern)]
        return int(await self.client.delete(*keys)) if keys else 0
