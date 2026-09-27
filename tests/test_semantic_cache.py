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
