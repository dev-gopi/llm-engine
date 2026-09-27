import asyncio

import httpx
import pytest
import torch

from serving.api import ServingSettings, create_app
from serving.auth import OIDCAuthenticator, OIDCConfig, RBACPolicy, TenantQuotaLimiter
from serving.idempotency import IdempotencyConflict, IdempotencyStore
from serving.production_semantic_cache import (
    ProductionCacheRequest,
    ProductionSemanticCache,
    TenantQuotaManager,
)
from serving.redis_semantic_cache import RedisCacheConfig, RedisSemanticCacheBackend
from serving.runtime import BackendGeneration, BackendStreamEvent
from serving.schemas import FinishReason, GenerateRequest
from serving.semantic_cache_policy import CacheQuota, ThresholdPolicy
from serving.webhooks import WebhookConfig, WebhookDelivery


class FakeBackend:
    ready = True

    async def generate(self, req):
        return BackendGeneration(
            f"ok:{req.prompt}:{req.tenant_id}", 2, 3, FinishReason.STOP
        )

    async def stream(self, req):
        yield BackendStreamEvent(token="ok", prompt_tokens=2, completion_tokens=1)
        yield BackendStreamEvent(
            finish_reason=FinishReason.STOP, prompt_tokens=2, completion_tokens=1
        )


def settings(tmp_path, **kw):
    base = dict(
        model_name="gopi-test",
        bot_name="Gopi",
        platform_db_path=str(tmp_path / "platform.sqlite"),
        platform_files_dir=str(tmp_path / "files"),
    )
    base.update(kw)
    return ServingSettings(**base)


async def call(app, method, path, **kwargs):
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://test",
        ) as c:
            return await c.request(method, path, **kwargs)


def req(app, method, path, **kwargs):
    return asyncio.run(call(app, method, path, **kwargs))


def test_files_vector_batch_finetune_lifecycle(tmp_path):
    app = create_app(FakeBackend(), settings=settings(tmp_path))
    uploaded = req(
        app,
        "POST",
        "/v1/files",
        files={"file": ("train.jsonl", b'{"x":1}\n', "application/jsonl")},
        data={"purpose": "fine-tune"},
    )
    assert uploaded.status_code == 200, uploaded.text
    fid = uploaded.json()["id"]
    assert req(app, "GET", "/v1/files").json()["data"][0]["id"] == fid
    vs = req(app, "POST", "/v1/vector_stores", json={"name": "docs"}).json()
    attached = req(
        app, "POST", f"/v1/vector_stores/{vs['id']}/files", json={"file_id": fid}
    )
    assert attached.status_code == 200
    batch = req(
        app,
        "POST",
        "/v1/batches",
        json={
            "input_file_id": fid,
            "endpoint": "/v1/responses",
            "completion_window": "24h",
        },
    )
    assert batch.status_code == 200 and batch.json()["status"] == "validating"
    ft = req(
        app,
        "POST",
        "/v1/fine_tuning/jobs",
        json={"training_file": fid, "model": "gopi-test"},
    )
    assert ft.status_code == 200 and ft.json()["status"] == "queued"
    assert req(app, "DELETE", f"/v1/files/{fid}").json()["deleted"]


def test_oidc_hs256_and_tenant_binding(tmp_path):
    jwt = pytest.importorskip("jwt")
    token = jwt.encode(
        {
            "sub": "alice",
            "tenant_id": "acme",
            "roles": ["tenant_admin"],
            "aud": "gopi",
            "iss": "https://issuer",
        },
        "secret",
        algorithm="HS256",
    )
    app = create_app(
        FakeBackend(),
        settings=settings(
            tmp_path,
            oidc_enabled=True,
            oidc_issuer="https://issuer",
            oidc_audience="gopi",
            oidc_hs256_secret="secret",
        ),
    )
    response = req(
        app,
        "POST",
        "/v1/generate",
        headers={"Authorization": f"Bearer {token}", "X-Tenant-ID": "evil"},
        json={"prompt": "hello"},
    )
    assert response.status_code == 200
    assert response.json()["text"].endswith(":acme")
    assert req(app, "POST", "/v1/generate", json={"prompt": "hello"}).status_code == 401


def test_rbac_and_tenant_quota():
    auth = OIDCAuthenticator(OIDCConfig())
    limiter = TenantQuotaLimiter(2)
    assert (
        limiter.allow("a", now=0)
        and limiter.allow("a", now=1)
        and not limiter.allow("a", now=2)
    )
    from serving.auth import AuthPrincipal

    assert RBACPolicy.allowed(AuthPrincipal("x", roles=("tenant_admin",)), "developer")
    assert not RBACPolicy.allowed(AuthPrincipal("x", roles=("user",)), "admin")


def test_idempotency_conflict(tmp_path):
    store = IdempotencyStore(tmp_path / "idempotency.sqlite")
    h = store.request_hash({"a": 1})
    store.put("t", "/x", "k", h, 200, {"ok": True})
    assert store.get("t", "/x", "k", h) == (200, {"ok": True})
    with pytest.raises(IdempotencyConflict):
        store.get("t", "/x", "k", store.request_hash({"a": 2}))


class FakeRedis:
    def __init__(self):
        self.data = {}
        self.locks = {}

    async def get(self, k):
        return self.data.get(k)

    async def set(self, k, v, ex=None, nx=False, px=None):
        if nx and k in self.data:
            return False
        self.data[k] = v
        return True

    async def delete(self, *keys):
        return sum(self.data.pop(k, None) is not None for k in keys)

    async def eval(self, script, n, key, token):
        return await self.delete(key)

    async def scan_iter(self, match=None, count=None):
        import fnmatch

        for k in list(self.data):
            if fnmatch.fnmatch(k, match):
                yield k


def test_distributed_semantic_vector_lookup_and_quota():
    async def run():
        redis = RedisSemanticCacheBackend(RedisCacheConfig(), client=FakeRedis())
        cache = ProductionSemanticCache(
            redis_backend=redis,
            quota_manager=TenantQuotaManager(CacheQuota(max_entries=1)),
            threshold_policy=ThresholdPolicy(default=0.1),
        )
        ctx = ProductionCacheRequest(tenant="t")
        a = GenerateRequest(prompt="hello world", temperature=0, tenant_id="t")
        result = BackendGeneration("answer", 2, 1, FinishReason.STOP)
        key = cache.key(a, tenant="t")
        assert await cache.store(a, ctx, key, result)
        b = GenerateRequest(prompt="hello world!", temperature=0, tenant_id="t")
        found, score = await cache.lookup(b, ctx, cache.key(b, tenant="t"))
        assert found is not None and found.text == "answer" and score is not None
        c = GenerateRequest(
            prompt="another unrelated prompt", temperature=0, tenant_id="t"
        )
        assert not await cache.store(c, ctx, cache.key(c, tenant="t"), result)

    asyncio.run(run())


def test_distributed_stream_cache_replay():
    async def run():
        from serving.runtime import ServingRuntime

        redis = RedisSemanticCacheBackend(RedisCacheConfig(), client=FakeRedis())
        cache = ProductionSemanticCache(
            redis_backend=redis, threshold_policy=ThresholdPolicy(default=0.1)
        )
        backend = FakeBackend()
        runtime = ServingRuntime(backend, production_semantic_cache=cache)
        r = GenerateRequest(prompt="stream me", temperature=0, tenant_id="t")
        first = [e async for e in runtime.stream(r)]
        second = [e async for e in runtime.stream(r)]
        assert any(e.token == "ok" for e in first)
        assert any(e.token == "ok" for e in second)

    asyncio.run(run())


def test_lora_admin_lifecycle(tmp_path):
    class G:
        def __init__(self):
            self.states = []

        def swap_lora_adapter(self, state):
            self.states.append(state)

    backend = FakeBackend()
    backend.generator = G()
    root = tmp_path / "adapters"
    root.mkdir()
    torch.save({"adapter": {"x": torch.tensor([1.0])}}, root / "a.pt")
    app = create_app(
        backend,
        settings=settings(tmp_path, admin_api_key="adm", lora_adapter_root=str(root)),
    )
    h = {"X-Admin-API-Key": "adm"}
    pub = req(
        app,
        "POST",
        "/admin/lora/adapters",
        headers=h,
        json={
            "tenant_id": "t",
            "adapter_id": "a",
            "version": "1",
            "checkpoint_path": "a.pt",
        },
    )
    assert pub.status_code == 200, pub.text
    act = req(app, "POST", "/admin/lora/adapters/a/activate?tenant_id=t", headers=h)
    assert act.status_code == 200 and backend.generator.states
    de = req(app, "POST", "/admin/lora/deactivate?tenant_id=t", headers=h)
    assert de.status_code == 200 and backend.generator.states[-1] is None


def test_grammar_request_contract():
    req = GenerateRequest(prompt="x", grammar='start: "yes"')
    assert req.grammar_start == "start"
    with pytest.raises(Exception):
        GenerateRequest(
            prompt="x", grammar='start: "yes"', response_format={"type": "json_schema"}
        )


def test_webhook_signature_and_dead_letter(tmp_path, monkeypatch):
    delivery = WebhookDelivery(
        WebhookConfig(
            "https://example.invalid/hook",
            "secret",
            max_attempts=1,
            dead_letter_path=str(tmp_path / "dlq.jsonl"),
        )
    )
    headers = delivery._headers(b"{}", 123)
    assert headers["X-Gopi-Signature"].startswith("v1=")
