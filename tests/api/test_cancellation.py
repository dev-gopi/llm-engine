import asyncio
import pytest
from serving.api import _generate_with_disconnect
from serving.runtime import ServingRuntime, BackendGeneration
from serving.schemas import GenerateRequest, FinishReason

class SlowBackend:
    ready=True
    async def generate(self, request):
        await asyncio.sleep(1)
        return BackendGeneration("ok",1,1,FinishReason.STOP)
    async def stream(self, request):
        yield

class DisconnectingRequest:
    def __init__(self): self.calls=0
    async def is_disconnected(self):
        self.calls += 1
        return self.calls >= 2

@pytest.mark.asyncio
async def test_disconnect_cancels_active_generation():
    runtime=ServingRuntime(SlowBackend(),generation_timeout_seconds=5)
    await runtime.startup()
    with pytest.raises(asyncio.CancelledError):
        await _generate_with_disconnect(runtime, DisconnectingRequest(), GenerateRequest(prompt="x"))
    assert runtime.active_requests == 0

def test_explicit_cancel_endpoint_returns_contract():
    from fastapi.testclient import TestClient
    from serving.api import create_app, ServingSettings
    from tests.test_serving import FakeBackend
    settings=ServingSettings(model_name="gopi-test",bot_name="Gopi",allowed_hosts=("testserver","test","localhost","127.0.0.1"))
    with TestClient(create_app(FakeBackend(),settings=settings)) as client:
        response=client.post('/v1/requests/req_123/cancel')
    assert response.status_code==200 and response.json()=={"request_id":"req_123","cancelled":False}
