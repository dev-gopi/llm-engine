"""Production semantic-cache orchestration primitives.

This module layers distributed coordination, tenant quotas, privacy controls,
freshness fingerprints, negative caching, warming, selective purge, metrics,
explainability and stale-while-revalidate over the existing local cache.
"""
from __future__ import annotations

import asyncio
import hashlib
import inspect
import time
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Mapping

from .redis_semantic_cache import RedisSemanticCacheBackend
from .runtime import BackendGeneration
from .semantic_cache_policy import (
    CacheDebugInfo,
    CacheFreshness,
    CachePrivacyPolicy,
    CacheQuota,
    NegativeCacheEntry,
    ThresholdPolicy,
)
from .schemas import FinishReason, GenerateRequest, OpenAIToolCall


@dataclass(frozen=True)
class ProductionCacheRequest:
    tenant: str = "default"
    user: str | None = None
    route: str | None = None
    model: str | None = None
    task: str | None = None
    freshness: CacheFreshness | None = None
    privacy: CachePrivacyPolicy = CachePrivacyPolicy()


class TenantQuotaManager:
    def __init__(self, default: CacheQuota = CacheQuota(), quotas: Mapping[str, CacheQuota] | None = None) -> None:
        self.default = default
        self.quotas = dict(quotas or {})
        self._counts: dict[str, int] = {}

    def quota_for(self, tenant: str) -> CacheQuota:
        return self.quotas.get(tenant, self.default)

    def reserve(self, tenant: str) -> bool:
        quota = self.quota_for(tenant)
        if quota.max_entries <= 0:
            return True
        current = self._counts.get(tenant, 0)
        if current >= quota.max_entries:
            return False
        self._counts[tenant] = current + 1
        return True

    def release(self, tenant: str) -> None:
        self._counts[tenant] = max(0, self._counts.get(tenant, 0) - 1)


class ProductionSemanticCache:
    """Backend-neutral policy layer with optional Redis distribution."""

    def __init__(
        self,
        *,
        redis_backend: RedisSemanticCacheBackend | None = None,
        quota_manager: TenantQuotaManager | None = None,
        threshold_policy: ThresholdPolicy | None = None,
        negative_ttl_seconds: float = 15.0,
        stale_while_revalidate_seconds: float = 30.0,
    ) -> None:
        if negative_ttl_seconds <= 0 or stale_while_revalidate_seconds <= 0:
            raise ValueError("negative and stale-while-revalidate TTLs must be positive")
        self.redis = redis_backend
        self.quotas = quota_manager or TenantQuotaManager()
        self.thresholds = threshold_policy or ThresholdPolicy()
        self.negative_ttl_seconds = negative_ttl_seconds
        self.swr_seconds = stale_while_revalidate_seconds
        self._negative: dict[str, NegativeCacheEntry] = {}
        self._metrics: dict[str, float] = {
            "lookups": 0, "hits": 0, "misses": 0, "negative_hits": 0,
            "stores": 0, "bypasses": 0, "singleflight_waits": 0,
            "refreshes": 0, "refresh_failures": 0, "tokens_saved": 0,
            "latency_saved_ms": 0.0, "similarity_sum": 0.0, "similarity_count": 0,
        }

    @staticmethod
    def key(request: GenerateRequest, *, tenant: str, freshness: CacheFreshness | None = None) -> str:
        payload = request.model_dump(mode="json")
        payload.pop("prompt", None)
        payload["tenant"] = tenant
        payload["freshness"] = freshness.value if freshness else ""
        return hashlib.sha256(str(sorted(payload.items())).encode()).hexdigest()

    def policy(
        self,
        request: GenerateRequest,
        context: ProductionCacheRequest,
    ) -> tuple[bool, str | None]:
        bypass, reason = context.privacy.should_bypass(request.prompt)
        if bypass:
            self._metrics["bypasses"] += 1
            return False, reason
        if request.cache_control == "no-store":
            self._metrics["bypasses"] += 1
            return False, "no_store"
        dynamic = (request.session_id or request.tools or request.chat_tools or request.mcp or request.attachments or request.rag or request.web_search)
        if dynamic and context.freshness is None:
            self._metrics["bypasses"] += 1
            return False, "missing_freshness_fingerprint"
        return True, None

    def threshold(self, context: ProductionCacheRequest) -> float:
        return self.thresholds.resolve(route=context.route, model=context.model, task=context.task)

    def metrics(self) -> dict[str, float]:
        result = dict(self._metrics)
        result["hit_rate"] = result["hits"] / result["lookups"] if result["lookups"] else 0.0
        result["mean_similarity"] = result["similarity_sum"] / result["similarity_count"] if result["similarity_count"] else 0.0
        return result

    def explain_miss(self, *, namespace: str, threshold: float, reason: str | None = None) -> CacheDebugInfo:
        return CacheDebugInfo(False, "miss", None, threshold, namespace, bypass_reason=reason)

    def record_similarity(self, similarity: float) -> None:
        self._metrics["similarity_sum"] += float(similarity)
        self._metrics["similarity_count"] += 1

    def record_hit(self, *, similarity: float | None = None, tokens_saved: int = 0, latency_saved_ms: float = 0.0) -> None:
        self._metrics["hits"] += 1
        self._metrics["tokens_saved"] += max(0, int(tokens_saved))
        self._metrics["latency_saved_ms"] += max(0.0, float(latency_saved_ms))
        if similarity is not None:
            self.record_similarity(similarity)

    def record_lookup(self, hit: bool) -> None:
        self._metrics["lookups"] += 1
        if not hit:
            self._metrics["misses"] += 1

    def negative_put(self, key: str, error_type: str, message: str) -> None:
        self._negative[key] = NegativeCacheEntry.create(key, error_type, message, self.negative_ttl_seconds)

    def negative_get(self, key: str) -> NegativeCacheEntry | None:
        entry = self._negative.get(key)
        if entry is None:
            return None
        if not entry.active():
            self._negative.pop(key, None)
            return None
        self._metrics["negative_hits"] += 1
        return entry

    async def singleflight(
        self,
        key: str,
        tenant: str,
        work: Callable[[], Awaitable[BackendGeneration]],
        *,
        wait_timeout: float = 30.0,
    ) -> BackendGeneration:
        if self.redis is None:
            return await work()
        token = uuid.uuid4().hex
        if await self.redis.acquire_singleflight(tenant, key, token):
            try:
                return await work()
            finally:
                await self.redis.release_singleflight(tenant, key, token)
        self._metrics["singleflight_waits"] += 1
        deadline = time.monotonic() + wait_timeout
        while time.monotonic() < deadline:
            cached = await self.redis.get(tenant, key)
            if cached is not None:
                payload = dict(cached)
                payload["finish_reason"] = FinishReason(payload.get("finish_reason", FinishReason.STOP))
                payload["tool_calls"] = tuple(
                    item if isinstance(item, OpenAIToolCall) else OpenAIToolCall.model_validate(item)
                    for item in payload.get("tool_calls", ())
                )
                payload["logprobs"] = tuple(payload.get("logprobs", ()))
                return BackendGeneration(**payload)
            await asyncio.sleep(0.05)
        return await work()

    async def refresh(self, refresh_fn: Callable[[], Awaitable[Any]]) -> None:
        self._metrics["refreshes"] += 1
        try:
            result = refresh_fn()
            if inspect.isawaitable(result):
                await result
        except Exception:
            self._metrics["refresh_failures"] += 1

    async def warm(self, tenant: str, entries: list[tuple[str, dict[str, Any], float]]) -> int:
        if self.redis is None:
            return 0
        return await self.redis.warm(tenant, entries)

    async def purge(self, tenant: str, *, exact_key: str | None = None) -> int:
        if self.redis is None:
            return 0
        return await self.redis.purge(tenant, exact_key=exact_key)
