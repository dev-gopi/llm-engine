from tests.asgi_client import ASGIClient

from serving.api import ServingSettings, create_app
from serving.runtime import BackendGeneration, BackendStreamEvent
from serving.schemas import FinishReason


class Backend:
    ready = True
    async def startup(self): pass
    async def shutdown(self): pass
    async def generate(self, request):
        return BackendGeneration(text='{"answer":"ok"}' if isinstance(request.response_format, dict) else "ok", prompt_tokens=2, completion_tokens=2, finish_reason=FinishReason.STOP)
    async def stream(self, request):
        yield BackendStreamEvent(token="ok", completion_tokens=1)
        yield BackendStreamEvent(finish_reason=FinishReason.STOP, prompt_tokens=2, completion_tokens=1)


def make_client():
    settings = ServingSettings(model_name="gopi-test", bot_name="Gopi", api_key="secret", allowed_hosts=("testserver", "test", "localhost", "127.0.0.1"))
    return ASGIClient(create_app(Backend(), settings=settings))


def test_openai_conformance_core_contract():
    with make_client() as client:
        headers = {"Authorization": "Bearer secret"}
        response = client.post("/v1/chat/completions", headers=headers, json={"model":"gopi-test","messages":[{"role":"user","content":"hello"}]})
        assert response.status_code == 200
        payload = response.json()
        assert payload["object"] == "chat.completion"
        assert payload["usage"]["total_tokens"] == payload["usage"]["prompt_tokens"] + payload["usage"]["completion_tokens"]
        assert client.post("/v1/chat/completions", headers=headers, json={"model":"gopi-test","messages":[]}).status_code == 422


def test_openai_streaming_usage_models_auth_and_errors():
    with make_client() as client:
        headers = {"Authorization": "Bearer secret"}
        streamed = client.post("/v1/chat/completions", headers=headers, json={"model":"gopi-test","messages":[{"role":"user","content":"x"}],"stream":True})
        assert streamed.status_code == 200 and "data:" in streamed.text
        models = client.get("/v1/models", headers=headers)
        assert models.status_code == 200 and models.json()["data"][0]["id"] == "gopi-test"
        assert client.get("/v1/models", headers={"Authorization":"Bearer wrong"}).status_code == 401


def test_openai_generation_parameters_are_exposed_and_validated():
    with make_client() as client:
        headers = {"Authorization": "Bearer secret"}
        response = client.post("/v1/chat/completions", headers=headers, json={
            "model":"gopi-test","messages":[{"role":"user","content":"x"}],
            "temperature":0.2,"top_p":0.8,"top_k":10,"min_p":0.1,"seed":42,
            "max_tokens":4,"stop":["END"],"repetition_penalty":1.05,
        })
        assert response.status_code == 200
        unsupported = client.post("/v1/chat/completions", headers=headers, json={
            "model":"gopi-test","messages":[{"role":"user","content":"x"}],"presence_penalty":0.2,
        })
        assert unsupported.status_code == 422


def test_responses_api_and_embeddings_are_discoverable():
    with make_client() as client:
        headers = {"Authorization": "Bearer secret"}
        response = client.post("/v1/responses", headers=headers, json={"model":"gopi-test","input":"hello"})
        assert response.status_code == 200
        embedding = client.post("/v1/embeddings", headers=headers, json={"model":"gopi-embedding-hash","input":["a","b"]})
        assert embedding.status_code == 200
        assert len(embedding.json()["data"]) == 2
        assert embedding.json()["usage"]["total_tokens"] > 0
