import asyncio
import time

import torch

from embeddings import EmbeddingService
from serving.runtime import BackendGeneration, BackendStreamEvent, ServingRuntime
from serving.schemas import FinishReason, GenerateRequest
from serving.semantic_cache import SemanticCacheConfig, SemanticResponseCache


class MeaningEncoder:
    dimension = 3

    def encode(self, texts):
        rows = []
        for text in texts:
            normalized = text.lower()
            if "capital" in normalized and (
                "france" in normalized or "french" in normalized
            ):
                rows.append([1.0, 0.0, 0.0])
            elif "weather" in normalized:
                rows.append([0.0, 1.0, 0.0])
            else:
                rows.append([0.0, 0.0, 1.0])
        return torch.tensor(rows, dtype=torch.float32)


class CountingBackend:
    ready = True

    def __init__(self):
        self.generate_calls = 0
        self.stream_calls = 0

    async def startup(self):
        return None

    async def shutdown(self):
        return None

    async def generate(self, request):
        self.generate_calls += 1
        await asyncio.sleep(0.01)
        return BackendGeneration(
            text=f"answer:{request.prompt}",
            prompt_tokens=4,
            completion_tokens=2,
            finish_reason=FinishReason.STOP,
        )

    async def stream(self, request):
        self.stream_calls += 1
        yield BackendStreamEvent(token="answer:", prompt_tokens=4, completion_tokens=1)
        yield BackendStreamEvent(token=request.prompt, completion_tokens=2)
        yield BackendStreamEvent(
            finish_reason=FinishReason.STOP,
            prompt_tokens=4,
            completion_tokens=2,
        )


def make_cache(**overrides):
    values = dict(
        enabled=True,
        capacity=8,
        similarity_threshold=0.99,
        ttl_seconds=60.0,
        namespace="tests-v1",
    )
    values.update(overrides)
    return SemanticResponseCache(
        SemanticCacheConfig(**values),
        embedding_service=EmbeddingService(MeaningEncoder()),
    )


def deterministic(prompt, **overrides):
    values = dict(prompt=prompt, temperature=0.0, top_k=0, top_p=1.0)
    values.update(overrides)
    return GenerateRequest(**values)


def test_exact_and_semantic_hits_are_separate_and_safe():
    cache = make_cache()
    first = deterministic("What is the capital of France?")
    result = BackendGeneration("Paris", 6, 1, FinishReason.STOP)
    assert cache.lookup(first) is None
    assert cache.store(first, result)

    exact = cache.lookup(first)
    assert exact is not None and exact.text == "Paris"
    assert exact.cached_tokens == exact.prompt_tokens

    paraphrase = deterministic("Tell me the French capital")
    semantic = cache.lookup(paraphrase)
    assert semantic is not None and semantic.text == "Paris"

    metrics = cache.metrics()
    assert metrics["exact_hits"] == 1
    assert metrics["semantic_hits"] == 1
    assert metrics["misses"] == 1


def test_semantic_hit_requires_identical_generation_policy():
    cache = make_cache()
    source = deterministic("What is the capital of France?", max_tokens=32)
    cache.store(source, BackendGeneration("Paris", 6, 1, FinishReason.STOP))
    changed = deterministic("Tell me the French capital", max_tokens=64)
    assert cache.lookup(changed) is None


def test_dynamic_and_nondeterministic_requests_bypass_cache():
    cache = make_cache()
    assert not cache.eligible(GenerateRequest(prompt="hello", temperature=0.7))
    assert not cache.eligible(deterministic("hello", session_id="session-1"))
    assert not cache.eligible(deterministic("hello", rag=True))
    assert not cache.eligible(deterministic("hello", web_search=True))
    assert not cache.eligible(
        deterministic("hello", attachments=[{"name": "a.txt", "content": "x"}])
    )
    assert cache.lookup(GenerateRequest(prompt="hello", temperature=0.7)) is None
    assert cache.metrics()["bypasses"] == 1


def test_ttl_expiry_removes_entry():
    cache = make_cache(ttl_seconds=0.01)
    request = deterministic("What is the capital of France?")
    cache.store(request, BackendGeneration("Paris", 6, 1, FinishReason.STOP))
    time.sleep(0.02)
    assert cache.lookup(request) is None
    assert cache.metrics()["expired"] >= 1


def test_sqlite_cache_survives_process_recreation(tmp_path):
    path = tmp_path / "semantic-cache.sqlite"
    request = deterministic("What is the capital of France?")
    cache = make_cache(sqlite_path=str(path))
    cache.store(request, BackendGeneration("Paris", 6, 1, FinishReason.STOP))
    cache.close()

    restored = make_cache(sqlite_path=str(path))
    hit = restored.lookup(request)
    assert hit is not None and hit.text == "Paris"
    assert restored.purge() == 1
    restored.close()


def test_runtime_stampede_protection_generates_once_for_identical_requests():
    async def scenario():
        backend = CountingBackend()
        cache = make_cache()
        runtime = ServingRuntime(backend, max_concurrency=4, semantic_cache=cache)
        await runtime.startup()
        request = deterministic("What is the capital of France?")
        results = await asyncio.gather(*(runtime.generate(request) for _ in range(4)))
        metrics = runtime.metrics()
        await runtime.shutdown()
        return backend, results, metrics

    backend, results, metrics = asyncio.run(scenario())
    assert backend.generate_calls == 1
    assert [result.text for result in results] == [
        "answer:What is the capital of France?"
    ] * 4
    assert metrics["semantic_cache_exact_hits"] == 3


def test_runtime_stream_replays_cached_response_without_backend_stream_call():
    async def scenario():
        backend = CountingBackend()
        cache = make_cache()
        runtime = ServingRuntime(backend, semantic_cache=cache)
        await runtime.startup()
        request = deterministic("What is the capital of France?")
        first = [event async for event in runtime.stream(request)]
        second = [event async for event in runtime.stream(request)]
        await runtime.shutdown()
        return backend, first, second

    backend, first, second = asyncio.run(scenario())
    assert backend.stream_calls == 1
    assert (
        "".join(event.token for event in first)
        == "answer:What is the capital of France?"
    )
    assert (
        "".join(event.token for event in second)
        == "answer:What is the capital of France?"
    )
    assert second[-1].finish_reason == FinishReason.STOP


def test_api_admin_can_inspect_and_purge_semantic_cache():
    import httpx

    from serving.api import ServingSettings, create_app

    async def scenario():
        backend = CountingBackend()
        app = create_app(
            backend,
            settings=ServingSettings(
                model_name="gopi-test",
                bot_name="Gopi",
                max_concurrency=2,
                queue_timeout_seconds=0.1,
                generation_timeout_seconds=1.0,
                admin_api_key="admin-secret",
                semantic_cache_enabled=True,
                semantic_cache_similarity_threshold=0.999,
                semantic_cache_namespace="api-test",
            ),
        )
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                await client.post(
                    "/v1/generate", json={"prompt": "cache me", "temperature": 0.0}
                )
                headers = {"X-Admin-API-Key": "admin-secret"}
                status = await client.get("/admin/cache/semantic", headers=headers)
                purged = await client.delete("/admin/cache/semantic", headers=headers)
                return status, purged

    status, purged = asyncio.run(scenario())
    assert status.status_code == 200
    assert status.json()["enabled"] is True
    assert status.json()["stores"] == 1
    assert purged.status_code == 200
    assert purged.json()["purged"] == 1


from serving.production_semantic_cache import (
    ProductionCacheRequest,
    ProductionSemanticCache,
    TenantQuotaManager,
)
from serving.redis_semantic_cache import RedisCacheConfig, RedisSemanticCacheBackend
from serving.semantic_cache_policy import (
    CacheFreshness,
    CachePrivacyPolicy,
    CacheQuota,
    ThresholdPolicy,
)


def req_gen(prompt="hello", **kwargs):
    return GenerateRequest(prompt=prompt, temperature=0.0, top_k=0, top_p=1.0, **kwargs)


def test_privacy_and_no_store_are_conservative():
    policy = CachePrivacyPolicy()
    assert policy.should_bypass("my api_key=secret_1234567890123456")[0]
    assert policy.should_bypass("email me at test@example.com")[0]
    assert not policy.should_bypass("What is 2+2?")[0]


def test_dynamic_request_requires_freshness():
    cache = ProductionSemanticCache()
    context = ProductionCacheRequest()
    ok, reason = cache.policy(req_gen("hello", rag=True), context)
    assert not ok and reason == "missing_freshness_fingerprint"
    context = ProductionCacheRequest(
        freshness=CacheFreshness.from_mapping({"rag": "v2"})
    )
    ok, reason = cache.policy(req_gen("hello", rag=True), context)
    assert ok and reason is None


def test_threshold_resolution_and_metrics():
    cache = ProductionSemanticCache(
        threshold_policy=ThresholdPolicy(default=0.98, routes={"/chat": 0.95})
    )
    assert cache.threshold(ProductionCacheRequest(route="/chat")) == 0.95
    cache.record_lookup(True)
    cache.record_hit(similarity=0.97, tokens_saved=10, latency_saved_ms=20)
    assert cache.metrics()["hit_rate"] == 1.0
    assert cache.metrics()["tokens_saved"] == 10
    assert cache.metrics()["mean_similarity"] == 0.97


def test_negative_cache_has_short_ttl():
    cache = ProductionSemanticCache(negative_ttl_seconds=0.01)
    cache.negative_put("k", "timeout", "backend timeout")
    assert cache.negative_get("k") is not None
    time.sleep(0.02)
    assert cache.negative_get("k") is None


def test_tenant_quota():
    quotas = TenantQuotaManager(default=CacheQuota(max_entries=1))
    assert quotas.reserve("a")
    assert not quotas.reserve("a")
    quotas.release("a")
    assert quotas.reserve("a")


def test_request_key_changes_with_freshness_and_tenant():
    a = ProductionSemanticCache.key(req_gen("hello"), tenant="a")
    b = ProductionSemanticCache.key(req_gen("hello"), tenant="b")
    c = ProductionSemanticCache.key(
        req_gen("hello"), tenant="a", freshness=CacheFreshness("x")
    )
    assert len({a, b, c}) == 3


def test_no_store_field_is_backward_compatible():
    request = req_gen("hello", cache_control="no-store")
    cache = ProductionSemanticCache()
    ok, reason = cache.policy(request, ProductionCacheRequest())
    assert not ok and reason == "no_store"


class FakeRedis:
    def __init__(self):
        self.data = {}
        self.locks = {}

    async def get(self, key):
        return self.data.get(key)

    async def set(self, key, value, ex=None, nx=False, px=None):
        if nx and key in self.locks:
            return False
        if nx:
            self.locks[key] = value
            return True
        self.data[key] = value
        return True

    async def delete(self, key):
        self.data.pop(key, None)
        self.locks.pop(key, None)
        return 1

    async def eval(self, script, count, key, token):
        if self.locks.get(key) == token:
            del self.locks[key]
            return 1
        return 0

    async def scan_iter(self, match=None):
        for key in list(self.data):
            yield key


def test_redis_backend_and_singleflight_contract():
    async def scenario():
        backend = RedisSemanticCacheBackend(RedisCacheConfig(), client=FakeRedis())
        await backend.put("tenant", "key", {"text": "ok"}, 30)
        assert (await backend.get("tenant", "key"))["text"] == "ok"
        cache = ProductionSemanticCache(redis_backend=backend)
        calls = 0

        async def work():
            nonlocal calls
            calls += 1
            return __import__(
                "serving.runtime", fromlist=["BackendGeneration"]
            ).BackendGeneration(
                "ok",
                1,
                1,
                __import__(
                    "serving.schemas", fromlist=["FinishReason"]
                ).FinishReason.STOP,
            )

        result = await cache.singleflight("key", "tenant", work)
        assert result.text == "ok" and calls == 1

    asyncio.run(scenario())


def test_offline_false_positive_and_answer_drift_evaluation():
    from evaluation.semantic_cache import SemanticCacheEvalCase, evaluate_semantic_cache

    report = evaluate_semantic_cache(
        [
            SemanticCacheEvalCase("q1", "q1b", True, True, 0.99),
            SemanticCacheEvalCase("q2", "q2b", False, False, 0.99),
            SemanticCacheEvalCase("q3", "q3b", True, False, 0.98),
            SemanticCacheEvalCase("q4", "q4b", False, False, 0.50),
        ],
        threshold=0.95,
    )
    assert report.cases == 4
    assert report.false_positive_rate == 1 / 3
    assert report.answer_drift_rate == 1 / 3
