import json

from tests.asgi_client import ASGIClient
from tests.test_serving import FakeBackend
from serving.api import ServingSettings, create_app
from serving.runtime import BackendGeneration, BackendStreamEvent
from serving.schemas import (
    FinishReason,
    OpenAIChatCompletionRequest,
    OpenAIToolCall,
    OpenAIToolFunctionCall,
)


def _settings():
    return ServingSettings(
        model_name="gopi-test",
        bot_name="Gopi",
        allowed_hosts=("testserver", "test", "localhost", "127.0.0.1"),
    )


def _tool():
    return {
        "type": "function",
        "function": {
            "name": "weather",
            "description": "Read weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "additionalProperties": False,
            },
        },
    }


def test_multimodal_chat_schema_preserves_openai_content_parts():
    request = OpenAIChatCompletionRequest.model_validate({
        "model": "gopi-test",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "What is shown?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,YQ=="}},
            ],
        }],
    })
    generation = request.generation_request("gopi-test")
    assert generation.prompt == "What is shown?\n[image]"
    assert generation._chat_messages[0]["content"][1]["type"] == "image_url"
    assert generation._chat_messages[0]["content"][1]["image_url"]["url"].startswith("data:image/png")


def test_chat_completion_returns_reasoning_and_standard_tool_calls():
    call = OpenAIToolCall(
        id="call_weather",
        function=OpenAIToolFunctionCall(name="weather", arguments='{"city":"Kolkata"}'),
    )

    class FeatureBackend(FakeBackend):
        supports_tool_calling = True
        supports_reasoning = True

        async def generate(self, request):
            assert request.reasoning_effort == "medium"
            assert request.chat_tools[0].function.name == "weather"
            return BackendGeneration(
                text="",
                prompt_tokens=12,
                completion_tokens=7,
                finish_reason=FinishReason.TOOL_CALLS,
                reasoning_tokens=3,
                reasoning_content="Need the weather tool.",
                tool_calls=(call,),
            )

    payload = {
        "model": "gopi-test",
        "messages": [{"role": "user", "content": "Weather in Kolkata?"}],
        "tools": [_tool()],
        "tool_choice": "required",
        "reasoning_effort": "medium",
    }
    with ASGIClient(create_app(FeatureBackend(), settings=_settings())) as client:
        response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    body = response.json()
    choice = body["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] is None
    assert choice["message"]["reasoning_content"] == "Need the weather tool."
    assert choice["message"]["tool_calls"][0]["function"] == {
        "name": "weather",
        "arguments": '{"city":"Kolkata"}',
    }
    assert body["usage"]["reasoning_tokens"] == 3


def test_chat_completion_streams_reasoning_tool_call_and_finish_reason():
    call = OpenAIToolCall(
        id="call_weather",
        function=OpenAIToolFunctionCall(name="weather", arguments='{"city":"Kolkata"}'),
    )

    class FeatureBackend(FakeBackend):
        supports_tool_calling = True
        supports_reasoning = True

        async def stream(self, request):
            yield BackendStreamEvent(reasoning_token="Need tool", prompt_tokens=4, completion_tokens=1)
            yield BackendStreamEvent(tool_calls=(call,), prompt_tokens=4, completion_tokens=2)
            yield BackendStreamEvent(
                finish_reason=FinishReason.TOOL_CALLS,
                prompt_tokens=4,
                completion_tokens=2,
            )

    payload = {
        "model": "gopi-test",
        "messages": [{"role": "user", "content": "Weather?"}],
        "tools": [_tool()],
        "tool_choice": "required",
        "reasoning_effort": "low",
        "stream": True,
    }
    with ASGIClient(create_app(FeatureBackend(), settings=_settings())) as client:
        response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    events = []
    for raw in response.text.splitlines():
        if not raw.startswith("data: ") or raw == "data: [DONE]":
            continue
        events.append(json.loads(raw[6:]))
    deltas = [item["choices"][0]["delta"] for item in events]
    assert any(delta.get("reasoning_content") == "Need tool" for delta in deltas)
    tool_delta = next(delta for delta in deltas if delta.get("tool_calls"))
    assert tool_delta["tool_calls"][0]["function"]["name"] == "weather"
    assert events[-1]["choices"][0]["finish_reason"] == "tool_calls"


def test_required_tool_failure_is_explicit_not_silently_text_answer():
    class FeatureBackend(FakeBackend):
        supports_tool_calling = True

        async def generate(self, request):
            return BackendGeneration(
                text="I could not construct the call.",
                prompt_tokens=3,
                completion_tokens=5,
                finish_reason=FinishReason.STOP,
                tool_call_error="model did not produce the required tool call",
            )

    payload = {
        "model": "gopi-test",
        "messages": [{"role": "user", "content": "Weather?"}],
        "tools": [_tool()],
        "tool_choice": "required",
    }
    with ASGIClient(create_app(FeatureBackend(), settings=_settings())) as client:
        response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["finish_reason"] == "error"
    assert body["choices"][0]["message"]["refusal"] == "model did not produce the required tool call"
    assert body["incomplete_details"] == {"reason": "tool_call_generation_failed"}


def test_native_backend_interpreter_combines_reasoning_and_forced_tool_protocol():
    from serving.backend import ConfiguredModelBackend

    class Tokenizer:
        @staticmethod
        def encode(value, *args, **kwargs):
            return value.split()

    dummy = type("DummyBackend", (), {"generator": type("Generator", (), {"tokenizer": Tokenizer()})()})()
    parsed = OpenAIChatCompletionRequest.model_validate({
        "model": "gopi-test",
        "messages": [{"role": "user", "content": "Weather?"}],
        "tools": [_tool()],
        "tool_choice": {"type": "function", "function": {"name": "weather"}},
        "reasoning_effort": "medium",
        "max_tokens": 32,
    }).generation_request("gopi-test")

    result = ConfiguredModelBackend._interpret_generated_text(
        dummy,
        parsed,
        '<thinking>I should use the weather function.</thinking>{"city":"Kolkata"}',
        "stop",
    )
    text, reasoning, calls, error, finish, *_ = result
    assert text == ""
    assert reasoning == "I should use the weather function."
    assert error is None
    assert finish is FinishReason.TOOL_CALLS
    assert calls[0].function.arguments == '{"city":"Kolkata"}'
