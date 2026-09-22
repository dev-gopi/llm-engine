import pytest

from api.usage import account_usage


def test_usage_accounting_is_consistent():
    usage=account_usage(100,40,cached_tokens=25,reasoning_tokens=10,prefix_cache_hit=True)
    assert usage.total_tokens==140
    assert usage.as_dict()["cached_tokens"]==25
    assert usage.as_dict()["reasoning_tokens"]==10
    assert usage.as_dict()["cache"]["prefix_cache_hit"] is True

def test_usage_accounting_rejects_impossible_counts():
    with pytest.raises(ValueError): account_usage(10,5,cached_tokens=11)
    with pytest.raises(ValueError): account_usage(10,5,reasoning_tokens=6)

def test_usage_cache_counts_are_bounded_by_prompt_tokens():
    usage = account_usage(12, 8, cached_tokens=12, reasoning_tokens=8, prefix_cache_miss=True)
    assert usage.total_tokens == 20
    assert usage.as_dict()["cache"]["prefix_cache_miss"] is True

def test_generate_response_includes_cache_and_reasoning_usage():
    from serving.api import ServingSettings, create_app
    from serving.runtime import BackendGeneration
    from serving.schemas import FinishReason
    from tests.asgi_client import ASGIClient

    class Backend:
        ready = True
        async def startup(self): pass
        async def shutdown(self): pass
        async def generate(self, request):
            return BackendGeneration("ok", 10, 4, FinishReason.STOP, cached_tokens=6, reasoning_tokens=2)
        async def stream(self, request):
            yield

    settings = ServingSettings(
        model_name="gopi-test", bot_name="Gopi",
        allowed_hosts=("testserver", "test", "localhost", "127.0.0.1"),
    )
    with ASGIClient(create_app(Backend(), settings=settings)) as client:
        payload = client.post("/v1/generate", json={"prompt": "hello"}).json()
        assert payload["usage"] == {
            "prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14,
            "cached_tokens": 6, "reasoning_tokens": 2,
        }
