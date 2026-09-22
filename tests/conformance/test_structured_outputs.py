from serving.api import ServingSettings, create_app
from serving.runtime import BackendGeneration, BackendStreamEvent
from serving.schemas import FinishReason
from tests.asgi_client import ASGIClient


class StructuredBackend:
    ready = True
    def __init__(self, text): self.text = text
    async def startup(self): pass
    async def shutdown(self): pass
    async def generate(self, request): return BackendGeneration(self.text, 2, 2, FinishReason.STOP)
    async def stream(self, request):
        yield BackendStreamEvent(token=self.text, completion_tokens=2)
        yield BackendStreamEvent(finish_reason=FinishReason.STOP, prompt_tokens=2, completion_tokens=2)


def spec():
    return {"type":"object","properties":{"user":{"type":"object","properties":{"id":{"type":"integer"}},"required":["id"],"additionalProperties":False},"items":{"type":"array","items":{"type":"string"}}},"required":["user","items"],"additionalProperties":False}


def test_structured_conformance_success_nested_array_required_and_closed():
    payload='{"user":{"id":1},"items":["a","b"]}'
    settings=ServingSettings(model_name="gopi-test",bot_name="Gopi",allowed_hosts=("testserver","test","localhost","127.0.0.1"))
    with ASGIClient(create_app(StructuredBackend(payload),settings=settings)) as client:
        response=client.post('/v1/chat/completions',json={"model":"gopi-test","messages":[{"role":"user","content":"x"}],"response_format":{"type":"json_schema","json_schema":{"name":"result","strict":True,"schema":spec()}}})
        assert response.status_code == 200
        import json
        value=json.loads(response.json()["choices"][0]["message"]["content"])
        assert value == {"user":{"id":1},"items":["a","b"]}


def test_structured_conformance_invalid_schema_incomplete_and_streaming():
    settings=ServingSettings(model_name="gopi-test",bot_name="Gopi",allowed_hosts=("testserver","test","localhost","127.0.0.1"))
    with ASGIClient(create_app(StructuredBackend('{"user":{"id":"bad"},"items":[]}'),settings=settings)) as client:
        base={"model":"gopi-test","messages":[{"role":"user","content":"x"}]}
        bad={**base,"response_format":{"type":"json_schema","json_schema":{"name":"result","strict":True,"schema":{"type":"wat"}}}}
        assert client.post('/v1/chat/completions',json=bad).status_code == 422
        invalid={**base,"response_format":{"type":"json_schema","json_schema":{"name":"result","strict":True,"schema":spec()}}}
        response=client.post('/v1/chat/completions',json=invalid)
        assert response.status_code == 200
        assert response.json()["incomplete_details"]["reason"] == "structured_output_validation_failed"

    with ASGIClient(create_app(StructuredBackend('{"user":{"id":1},"items":["ok"]}'),settings=settings)) as client:
        streamed=client.post('/v1/chat/completions',json={**base,"stream":True,"response_format":{"type":"json_schema","json_schema":{"name":"result","strict":True,"schema":spec()}}})
        assert streamed.status_code == 200
        assert "data:" in streamed.text
