import asyncio

import pytest
from fastapi.testclient import TestClient

from serving.api import ServingSettings, create_app
from serving.orchestration import ReloadableBackend
from serving.runtime import BackendGeneration
from serving.schemas import FinishReason


class LifecycleBackend:
    def __init__(self, name: str):
        self.name = name
        self.ready = False
        self.started = 0
        self.stopped = 0

    async def startup(self):
        self.started += 1
        self.ready = True

    async def shutdown(self):
        self.stopped += 1
        self.ready = False

    async def generate(self, request):
        return BackendGeneration(self.name, 1, 1, FinishReason.STOP)

    async def stream(self, request):
        yield


def test_admin_lifecycle_requires_dedicated_admin_key():
    class Backend:
        ready = True
        async def startup(self): pass
        async def shutdown(self): pass
        async def generate(self, request):
            return BackendGeneration("ok", 1, 1, FinishReason.STOP)
        async def stream(self, request):
            yield

    settings = ServingSettings(
        model_name="gopi-test",
        bot_name="Gopi",
        api_key="normal-key",
        admin_api_key="admin-key",
        allowed_hosts=("testserver", "test", "localhost", "127.0.0.1"),
    )
    with TestClient(create_app(Backend(), settings=settings)) as client:
        assert client.post("/admin/models/unload", headers={"Authorization": "Bearer normal-key"}).status_code == 403


def test_admin_load_unload_reload_endpoints_are_functional_and_audited():
    created = []

    def factory():
        backend = LifecycleBackend(f"model-{len(created) + 1}")
        created.append(backend)
        return backend, backend.name

    initial, version = factory()
    from serving.orchestration import ReloadableBackend
    wrapper = ReloadableBackend(initial, version=version, factory=factory)
    settings = ServingSettings(
        model_name="gopi-test",
        bot_name="Gopi",
        admin_api_key="admin-key",
        allowed_hosts=("testserver", "test", "localhost", "127.0.0.1"),
    )
    with TestClient(create_app(wrapper, settings=settings)) as client:
        headers = {"Authorization": "Bearer admin-key"}
        assert client.post("/admin/models/unload", headers=headers).json()["status"] == "unloaded"
        assert client.post("/admin/models/load", headers=headers).json()["status"] == "loaded"
        reloaded = client.post("/admin/models/reload", headers=headers)
        assert reloaded.status_code == 200
        assert reloaded.json()["status"] == "loaded"
        assert reloaded.json()["version"] == "model-3"


@pytest.mark.asyncio
async def test_reloadable_backend_unload_and_load():
    created = []

    def factory():
        backend = LifecycleBackend(f"model-{len(created) + 1}")
        created.append(backend)
        return backend, backend.name

    first = factory()
    wrapper = ReloadableBackend(first[0], version=first[1], factory=factory)
    await wrapper.startup()
    assert wrapper.ready
    await wrapper.unload()
    assert not wrapper.ready
    await wrapper.load_current()
    assert wrapper.ready
    assert wrapper.version == "model-2"
