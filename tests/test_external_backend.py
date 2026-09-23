import asyncio
import json

import httpx

from serving.external_backend import OpenAICompatibleBackend
from serving.schemas import FinishReason, GenerateRequest


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


def test_external_backend_closes_client_after_failed_startup(monkeypatch):
    clients = []

    class FailingClient:
        def __init__(self, **_kwargs):
            self.closed = False
            clients.append(self)

        async def get(self, _path):
            raise httpx.ConnectError("offline")

        async def aclose(self):
            self.closed = True

    monkeypatch.setattr("serving.external_backend.httpx.AsyncClient", FailingClient)

    async def scenario():
        backend = OpenAICompatibleBackend(base_url="http://local")
        await backend.startup()
        return backend

    backend = asyncio.run(scenario())
    assert not backend.ready
    assert backend._client is None
    assert clients[0].closed


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


def test_external_backend_forwards_multimodal_tools_and_parses_tool_reasoning():
    captured = {}

    def handler(request: httpx.Request):
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={
            "choices": [{
                "message": {
                    "content": None,
                    "reasoning_content": "Need a tool",
                    "tool_calls": [{
                        "id": "call_echo",
                        "type": "function",
                        "function": {"name": "echo", "arguments": '{"value":"image"}'},
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 8, "completion_tokens": 4},
        })

    async def scenario():
        from serving.schemas import OpenAITool

        backend = OpenAICompatibleBackend(
            base_url="http://local",
            model="external-test",
            supports_tool_calling=True,
            supports_vision=True,
            supports_reasoning=True,
        )
        backend._client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://local")
        backend._ready = True
        request = GenerateRequest(
            prompt="describe [image]",
            reasoning_effort="high",
            chat_tools=[OpenAITool.model_validate({
                "type": "function",
                "function": {
                    "name": "echo",
                    "parameters": {
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                        "required": ["value"],
                    },
                },
            })],
            tool_choice="required",
        )
        request._chat_messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "describe"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,YQ=="}},
            ],
        }]
        result = await backend.generate(request)
        await backend.shutdown()
        return result

    result = asyncio.run(scenario())
    assert captured["messages"][0]["content"][1]["type"] == "image_url"
    assert captured["tools"][0]["function"]["name"] == "echo"
    assert captured["tool_choice"] == "required"
    assert captured["reasoning_effort"] == "high"
    assert result.reasoning_content == "Need a tool"
    assert result.finish_reason is FinishReason.TOOL_CALLS
    assert result.tool_calls[0].function.name == "echo"


def test_external_backend_stream_assembles_reasoning_and_fragmented_tool_call():
    async def stream_bytes():
        yield b'data: {"choices":[{"delta":{"reasoning_content":"Need "},"finish_reason":null}]}\n\n'
        yield b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_","type":"function","function":{"name":"ec","arguments":"{\\\"value\\\":\\\""}}]},"finish_reason":null}]}\n\n'
        yield b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"echo","function":{"name":"ho","arguments":"ok\\\"}"}}]},"finish_reason":"tool_calls"}]}\n\n'
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
    assert any(event.reasoning_token == "Need " for event in events)
    assert events[-1].finish_reason is FinishReason.TOOL_CALLS
    assert events[-1].tool_calls[0].id == "call_echo"
    assert events[-1].tool_calls[0].function.name == "echo"
    assert events[-1].tool_calls[0].function.arguments == '{"value":"ok"}'
