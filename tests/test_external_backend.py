import asyncio
import json

import httpx

from serving.external_backend import OpenAICompatibleBackend
from serving.schemas import GenerateRequest, FinishReason


def test_external_backend_maps_openai_response_and_sampler_controls():
    captured = {}

    def handler(request: httpx.Request):
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "external-test"}]})
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 7, "completion_tokens": 2,
                "prompt_tokens_details": {"cached_tokens": 3},
                "completion_tokens_details": {"reasoning_tokens": 1},
            },
        })

    async def scenario():
        backend = OpenAICompatibleBackend(base_url="http://local", model="external-test", context_length=262144)
        backend._client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://local")
        backend._ready = True
        result = await backend.generate(GenerateRequest(
            prompt="hi", top_k=20, min_p=0.05, repetition_penalty=1.1,
            reasoning_effort="low", max_tokens=16,
        ))
        await backend.shutdown()
        return result

    result = asyncio.run(scenario())
    assert result.text == "hello"
    assert result.finish_reason is FinishReason.STOP
    assert result.cached_tokens == 3 and result.reasoning_tokens == 1
    assert captured["model"] == "external-test"
    assert captured["top_k"] == 20 and captured["min_p"] == 0.05
    assert captured["reasoning_effort"] == "low"


def test_external_backend_streams_sse_chunks():
    async def stream_bytes():
        yield b'data: {"choices":[{"delta":{"content":"exter"},"finish_reason":null}]}\n\n'
        yield b'data: {"choices":[{"delta":{"content":"nal"},"finish_reason":"stop"}]}\n\n'
        yield b'data: [DONE]\n\n'

    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            async for chunk in stream_bytes():
                yield chunk

    def handler(request: httpx.Request):
        return httpx.Response(200, stream=Stream())

    async def scenario():
        backend = OpenAICompatibleBackend(base_url="http://local", model="external-test")
        backend._client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://local")
        backend._ready = True
        events = [event async for event in backend.stream(GenerateRequest(prompt="hi"))]
        await backend.shutdown()
        return events

    events = asyncio.run(scenario())
    assert "".join(event.token for event in events) == "external"
    assert events[-1].finish_reason is FinishReason.STOP
    assert events[-1].completion_tokens == 2
