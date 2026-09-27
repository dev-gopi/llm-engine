"""Optional Redis-backed distributed semantic-cache primitives.

Redis stores metadata/vectors as JSON so the implementation works with a
plain Redis deployment.  The backend is intentionally conservative: vector
similarity is calculated client-side over namespace/policy candidates. A
future Redis Vector Search adapter can replace candidate discovery without
changing the public contract.
"""
from __future__ import annotations

import json
import math
import struct
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
    vector_index_enabled: bool = True
    vector_index_name: str = "llm_engine_semantic_cache_idx"
    vector_dimension: int | None = None

    def __post_init__(self) -> None:
        if self.lock_ttl_ms < 100:
            raise ValueError("lock_ttl_ms must be at least 100ms")


class RedisSemanticCacheBackend:
    """Async Redis backend with fail-open distributed single-flight."""

    def __init__(self, config: RedisCacheConfig, *, client: Any | None = None) -> None:
        self.config = config
        self.client = client
        self._vector_index_ready = False
        self._vector_index_failed = False

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


    def _vector_key(self, namespace: str, exact_key: str) -> str:
        return f"{self.config.key_prefix}:vector:{namespace}:{exact_key}"

    def _quota_members_key(self, namespace: str) -> str:
        return f"{self.config.key_prefix}:quota-members:{namespace}"

    async def _ensure_vector_index(self, dimension: int) -> bool:
        """Create a Redis Search HNSW index when Redis Stack/RediSearch is available.

        Plain Redis deployments remain supported: a failed index creation is memoized and
        semantic lookup falls back to the bounded scan implementation.
        """
        if not self.config.vector_index_enabled or self.client is None or self._vector_index_failed:
            return False
        if self._vector_index_ready:
            return True
        name = self.config.vector_index_name
        prefix = f"{self.config.key_prefix}:vector:"
        try:
            try:
                await self.client.execute_command("FT.INFO", name)
            except Exception:
                await self.client.execute_command(
                    "FT.CREATE", name, "ON", "HASH", "PREFIX", "1", prefix,
                    "SCHEMA",
                    "namespace", "TAG",
                    "policy", "TAG",
                    "expires_at", "NUMERIC",
                    "vector", "VECTOR", "HNSW", "6",
                    "TYPE", "FLOAT32", "DIM", str(int(dimension)), "DISTANCE_METRIC", "COSINE",
                )
            self._vector_index_ready = True
            return True
        except Exception:
            self._vector_index_failed = True
            return False

    @staticmethod
    def _vector_blob(vector: list[float]) -> bytes:
        return struct.pack("<" + "f" * len(vector), *[float(v) for v in vector])

    async def reserve_quota(self, namespace: str, exact_key: str, ttl_seconds: float, max_entries: int) -> bool:
        """Atomically enforce a cross-replica per-tenant entry quota using a Redis ZSET."""
        if self.client is None or max_entries <= 0:
            return True
        if not all(hasattr(self.client, name) for name in ("zrem", "zremrangebyscore", "zcard")):
            # Compatibility fallback for minimal Redis clients/test doubles.
            if await self.get(namespace, exact_key) is not None:
                return True
            return await self.count(namespace) < max_entries
        now = time.time(); expires = now + max(1.0, float(ttl_seconds))
        script = """
        redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', ARGV[1])
        if redis.call('ZSCORE', KEYS[1], ARGV[3]) then
          redis.call('ZADD', KEYS[1], ARGV[2], ARGV[3]); return 1
        end
        if redis.call('ZCARD', KEYS[1]) >= tonumber(ARGV[4]) then return 0 end
        redis.call('ZADD', KEYS[1], ARGV[2], ARGV[3])
        redis.call('EXPIRE', KEYS[1], math.max(1, math.ceil(tonumber(ARGV[5]))))
        return 1
        """
        return bool(await self.client.eval(script, 1, self._quota_members_key(namespace), now, expires, exact_key, int(max_entries), ttl_seconds))

    async def release_quota(self, namespace: str, exact_key: str) -> None:
        if self.client is not None and hasattr(self.client, "zrem"):
            await self.client.zrem(self._quota_members_key(namespace), exact_key)

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


    async def put_semantic(self, namespace: str, exact_key: str, payload: dict[str, Any], ttl_seconds: float, vector: list[float], *, policy: str | None = None) -> bool:
        enriched = dict(payload)
        enriched["_semantic_vector"] = [float(v) for v in vector]
        if policy is not None:
            enriched["_semantic_policy"] = policy
        stored = await self.put(namespace, exact_key, enriched, ttl_seconds)
        if stored and vector and await self._ensure_vector_index(len(vector)):
            try:
                key = self._vector_key(namespace, exact_key)
                await self.client.hset(key, mapping={
                    "namespace": namespace, "policy": policy or "",
                    "exact_key": exact_key, "expires_at": str(time.time() + ttl_seconds),
                    "vector": self._vector_blob(vector),
                })
                await self.client.expire(key, max(1, int(ttl_seconds)))
            except Exception:
                # Exact cache writes must not fail because the optional ANN side-index is unavailable.
                pass
        return stored

    async def find_similar(self, namespace: str, vector: list[float], threshold: float, *, limit: int = 512, policy: str | None = None) -> tuple[str, dict[str, Any], float] | None:
        """Return the best cosine-similar entry from Redis.

        This plain-Redis implementation scans bounded candidates and keeps the
        contract compatible with future Redis Vector Search/ANN acceleration.
        """
        if self.client is None:
            return None
        if not vector:
            return None
        if await self._ensure_vector_index(len(vector)):
            try:
                query = f"(@namespace:{{{namespace}}}" + (f" @policy:{{{policy}}}" if policy is not None else "") + ")=>[KNN 1 @vector $vec AS distance]"
                result = await self.client.execute_command(
                    "FT.SEARCH", self.config.vector_index_name, query,
                    "PARAMS", "2", "vec", self._vector_blob(vector),
                    "SORTBY", "distance", "ASC", "RETURN", "2", "exact_key", "distance",
                    "DIALECT", "2",
                )
                if isinstance(result, (list, tuple)) and len(result) >= 3 and int(result[0]) > 0:
                    fields = result[2]
                    if isinstance(fields, (list, tuple)):
                        values = {str(fields[i]): fields[i+1] for i in range(0, len(fields)-1, 2)}
                        exact = values.get("exact_key") or values.get("b'exact_key'")
                        distance = values.get("distance") or values.get("b'distance'")
                        if isinstance(exact, bytes): exact = exact.decode()
                        if isinstance(distance, bytes): distance = distance.decode()
                        score = 1.0 - float(distance)
                        if exact and score >= threshold:
                            payload = await self.get(namespace, str(exact))
                            if payload is not None:
                                return str(exact), payload, score
            except Exception:
                # RediSearch can disappear during failover/version changes; safely fall back.
                self._vector_index_ready = False
                self._vector_index_failed = True
        norm = math.sqrt(sum(float(x) * float(x) for x in vector))
        if norm == 0:
            return None
        pattern = f"{self.config.key_prefix}:entry:{namespace}:*"
        best = None
        seen = 0
        async for key in self.client.scan_iter(match=pattern, count=min(max(16, limit), 1000)):
            if seen >= limit:
                break
            seen += 1
            raw = await self.client.get(key)
            if not raw:
                continue
            try:
                payload = json.loads(raw)
                if policy is not None and payload.get("_semantic_policy") != policy:
                    continue
                other = payload.get("_semantic_vector")
                if not isinstance(other, list) or len(other) != len(vector):
                    continue
                other_norm = math.sqrt(sum(float(x) * float(x) for x in other))
                if other_norm == 0:
                    continue
                score = sum(float(a) * float(b) for a, b in zip(vector, other)) / (norm * other_norm)
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if score >= threshold and (best is None or score > best[2]):
                cleaned = dict(payload)
                cleaned.pop("_semantic_vector", None)
                cleaned.pop("_semantic_policy", None)
                exact_key = str(key).rsplit(":", 1)[-1]
                best = (exact_key, cleaned, float(score))
        return best

    async def count(self, namespace: str) -> int:
        if self.client is None:
            return 0
        pattern = f"{self.config.key_prefix}:entry:{namespace}:*"
        count = 0
        async for _ in self.client.scan_iter(match=pattern):
            count += 1
        return count

    async def delete(self, namespace: str, exact_key: str) -> bool:
        if self.client is None:
            return False
        deleted = bool(await self.client.delete(self._key(namespace, exact_key)))
        try:
            await self.client.delete(self._vector_key(namespace, exact_key))
            await self.release_quota(namespace, exact_key)
        except Exception:
            pass
        return deleted

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
        deleted = int(await self.client.delete(*keys)) if keys else 0
        # Keep ANN side-index and quota membership consistent with the primary cache.
        vector_pattern = f"{self.config.key_prefix}:vector:{namespace}:*"
        vector_keys = [key async for key in self.client.scan_iter(match=vector_pattern)]
        if vector_keys:
            await self.client.delete(*vector_keys)
        await self.client.delete(self._quota_members_key(namespace))
        return deleted
