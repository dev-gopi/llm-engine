import asyncio
import time

import pytest

from serving.production_semantic_cache import ProductionCacheRequest, ProductionSemanticCache, TenantQuotaManager
from serving.redis_semantic_cache import RedisCacheConfig, RedisSemanticCacheBackend
from serving.semantic_cache_policy import CacheFreshness, CachePrivacyPolicy, CacheQuota, ThresholdPolicy
from serving.schemas import GenerateRequest


def req(prompt="hello", **kwargs):
    return GenerateRequest(prompt=prompt, temperature=0.0, top_k=0, top_p=1.0, **kwargs)


def test_privacy_and_no_store_are_conservative():
    policy = CachePrivacyPolicy()
    assert policy.should_bypass("my api_key=secret_1234567890123456")[0]
    assert policy.should_bypass("email me at test@example.com")[0]
    assert not policy.should_bypass("What is 2+2?")[0]


def test_dynamic_request_requires_freshness():
    cache = ProductionSemanticCache()
    context = ProductionCacheRequest()
    ok, reason = cache.policy(req("hello", rag=True), context)
    assert not ok and reason == "missing_freshness_fingerprint"
    context = ProductionCacheRequest(freshness=CacheFreshness.from_mapping({"rag": "v2"}))
    ok, reason = cache.policy(req("hello", rag=True), context)
    assert ok and reason is None


def test_threshold_resolution_and_metrics():
    cache = ProductionSemanticCache(threshold_policy=ThresholdPolicy(default=.98, routes={"/chat": .95}))
    assert cache.threshold(ProductionCacheRequest(route="/chat")) == .95
    cache.record_lookup(True); cache.record_hit(similarity=.97, tokens_saved=10, latency_saved_ms=20)
    assert cache.metrics()["hit_rate"] == 1.0
    assert cache.metrics()["tokens_saved"] == 10
    assert cache.metrics()["mean_similarity"] == .97


def test_negative_cache_has_short_ttl():
    cache = ProductionSemanticCache(negative_ttl_seconds=.01)
    cache.negative_put("k", "timeout", "backend timeout")
    assert cache.negative_get("k") is not None
    time.sleep(.02)
    assert cache.negative_get("k") is None


def test_tenant_quota():
    quotas = TenantQuotaManager(default=CacheQuota(max_entries=1))
    assert quotas.reserve("a")
    assert not quotas.reserve("a")
    quotas.release("a")
    assert quotas.reserve("a")


def test_request_key_changes_with_freshness_and_tenant():
    a = ProductionSemanticCache.key(req("hello"), tenant="a")
    b = ProductionSemanticCache.key(req("hello"), tenant="b")
    c = ProductionSemanticCache.key(req("hello"), tenant="a", freshness=CacheFreshness("x"))
    assert len({a,b,c}) == 3


def test_no_store_field_is_backward_compatible():
    request = req("hello", cache_control="no-store")
    cache = ProductionSemanticCache()
    ok, reason = cache.policy(request, ProductionCacheRequest())
    assert not ok and reason == "no_store"


class FakeRedis:
    def __init__(self):
        self.data = {}
        self.locks = {}
    async def get(self, key): return self.data.get(key)
    async def set(self, key, value, ex=None, nx=False, px=None):
        if nx and key in self.locks: return False
        if nx:
            self.locks[key] = value; return True
        self.data[key] = value; return True
    async def delete(self, key):
        self.data.pop(key, None); self.locks.pop(key, None); return 1
    async def eval(self, script, count, key, token):
        if self.locks.get(key) == token:
            del self.locks[key]; return 1
        return 0
    async def scan_iter(self, match=None):
        for key in list(self.data):
            yield key


def test_redis_backend_and_singleflight_contract():
    async def scenario():
        backend = RedisSemanticCacheBackend(RedisCacheConfig(), client=FakeRedis())
        await backend.put("tenant", "key", {"text":"ok"}, 30)
        assert (await backend.get("tenant", "key"))["text"] == "ok"
        cache = ProductionSemanticCache(redis_backend=backend)
        calls = 0
        async def work():
            nonlocal calls
            calls += 1
            return __import__('serving.runtime', fromlist=['BackendGeneration']).BackendGeneration("ok", 1, 1, __import__('serving.schemas', fromlist=['FinishReason']).FinishReason.STOP)
        result = await cache.singleflight("key", "tenant", work)
        assert result.text == "ok" and calls == 1
    asyncio.run(scenario())


def test_offline_false_positive_and_answer_drift_evaluation():
    from evaluation.semantic_cache import SemanticCacheEvalCase, evaluate_semantic_cache
    report = evaluate_semantic_cache([
        SemanticCacheEvalCase("q1", "q1b", True, True, .99),
        SemanticCacheEvalCase("q2", "q2b", False, False, .99),
        SemanticCacheEvalCase("q3", "q3b", True, False, .98),
        SemanticCacheEvalCase("q4", "q4b", False, False, .50),
    ], threshold=.95)
    assert report.cases == 4
    assert report.false_positive_rate == 1/3
    assert report.answer_drift_rate == 1/3
