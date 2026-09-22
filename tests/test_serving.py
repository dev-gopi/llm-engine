import asyncio

import httpx
import pytest
from starlette.websockets import WebSocketDisconnect

from serving.api import ServingSettings, create_app
from serving.runtime import (
    BackendGeneration,
    BackendStreamEvent,
    GenerationTimeoutError,
    ServerBusyError,
    ServingRuntime,
    UnavailableBackend,
    InvalidGenerationRequestError,
)
from serving.schemas import FinishReason, GenerateRequest
from serving.websocket import generate_stream
from serving.backend import _load_mcp_config


class FakeBackend:
    def __init__(self):
        self.ready = True
        self.started = False
        self.stopped = False

    async def startup(self):
        self.started = True

    async def shutdown(self):
        self.stopped = True

    async def generate(self, request):
        return BackendGeneration(
            text=f"Gopi: {request.prompt}",
            prompt_tokens=2,
            completion_tokens=3,
            finish_reason=FinishReason.STOP,
        )

    async def stream(self, request):
        yield BackendStreamEvent(token="Go", token_id=10, prompt_tokens=2)
        yield BackendStreamEvent(token="pi", token_id=11)
        yield BackendStreamEvent(
            finish_reason=FinishReason.STOP,
            prompt_tokens=2,
            completion_tokens=2,
        )


def settings(**overrides):
    values = {
        "model_name": "gopi-test",
        "bot_name": "Gopi",
        "max_concurrency": 2,
        "queue_timeout_seconds": 0.1,
        "generation_timeout_seconds": 1.0,
    }
    values.update(overrides)
    return ServingSettings(**values)


async def send_request(app, method, path, **kwargs):
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.request(method, path, **kwargs)


def request(app, method, path, **kwargs):
    return asyncio.run(send_request(app, method, path, **kwargs))


def test_liveness_and_unavailable_readiness():
    app = create_app(UnavailableBackend(), settings=settings())
    live = request(app, "GET", "/health/live")
    ready = request(app, "GET", "/health/ready")
    assert live.status_code == 200
    assert live.json()["status"] == "ok"
    assert not live.json()["ready"]
    assert ready.status_code == 503
    assert ready.json()["status"] == "not_ready"


def test_api_key_rate_limit_and_metrics():
    configured = settings(api_key="secret", requests_per_minute=1)
    app = create_app(FakeBackend(), settings=configured)
    unauthorized = request(app, "POST", "/v1/generate", json={"prompt": "hello"})
    assert unauthorized.status_code == 401
    headers = {"Authorization": "Bearer secret"}
    accepted = request(app, "POST", "/v1/generate", json={"prompt": "hello"}, headers=headers)
    limited = request(app, "POST", "/v1/generate", json={"prompt": "again"}, headers=headers)
    assert accepted.status_code == 200
    assert limited.status_code == 429
    metrics = request(app, "GET", "/metrics").json()
    assert metrics["completed_requests"] == 1


def test_authenticated_audit_log_records_protected_operations_without_prompt_content():
    app = create_app(FakeBackend(), settings=settings(api_key="secret", audit_log_capacity=2))
    headers = {"Authorization": "Bearer secret", "X-Request-ID": "audit-123"}
    accepted = request(app, "POST", "/v1/generate", json={"prompt": "private prompt"}, headers=headers)
    assert accepted.status_code == 200
    events = request(app, "GET", "/v1/audit/events", headers=headers)
    assert events.status_code == 200
    event = events.json()["events"][0]
    assert event["request_id"] == "audit-123"
    assert event["path"] == "/v1/generate"
    assert event["status_code"] == 200
    assert "private prompt" not in str(event)


def test_session_memory_access_is_opt_in_authenticated_and_deletable(tmp_path):
    from inference.context import SQLiteSessionStore
    from tokenizer.bpe import BYTE_ENCODER
    from tokenizer.encoder import DEFAULT_SPECIAL_TOKENS, Tokenizer

    pieces = list(DEFAULT_SPECIAL_TOKENS) + list(BYTE_ENCODER.values())
    vocab = {piece: index for index, piece in enumerate(pieces)}
    tokenizer = Tokenizer(vocab, special_tokens={piece: vocab[piece] for piece in DEFAULT_SPECIAL_TOKENS})
    backend = FakeBackend()
    backend.sessions = SQLiteSessionStore(tmp_path / "sessions.sqlite", tokenizer, max_tokens=64, system_prompt="")
    memory = backend.sessions.load("alice")
    memory.add("user", "private preference")
    backend.sessions.save("alice", memory)
    headers = {"Authorization": "Bearer secret"}

    disabled = request(create_app(backend, settings=settings(api_key="secret")), "GET", "/v1/sessions/alice/memory", headers=headers)
    assert disabled.status_code == 403
    app = create_app(backend, settings=settings(api_key="secret", session_memory_enabled=True))
    loaded = request(app, "GET", "/v1/sessions/alice/memory", headers=headers)
    assert loaded.status_code == 200
    assert loaded.json()["messages"][-1]["content"] == "private preference"
    deleted = request(app, "DELETE", "/v1/sessions/alice/memory", headers=headers)
    assert deleted.status_code == 200 and deleted.json()["deleted"]
    assert request(app, "GET", "/v1/sessions/alice/memory", headers=headers).json()["messages"] == []


def test_session_context_api_reports_token_usage_and_compacts_history(tmp_path):
    from inference.context import SQLiteSessionStore
    from tokenizer.bpe import BYTE_ENCODER
    from tokenizer.encoder import DEFAULT_SPECIAL_TOKENS, Tokenizer

    pieces = list(DEFAULT_SPECIAL_TOKENS) + list(BYTE_ENCODER.values())
    vocab = {piece: index for index, piece in enumerate(pieces)}
    tokenizer = Tokenizer(vocab, special_tokens={piece: vocab[piece] for piece in DEFAULT_SPECIAL_TOKENS})
    backend = FakeBackend()
    backend.sessions = SQLiteSessionStore(tmp_path / "sessions.sqlite", tokenizer, max_tokens=32, system_prompt="system")
    memory = backend.sessions.load("alice")
    memory.add("user", "first message with enough words to be compacted")
    memory.add("assistant", "first response with enough words to be compacted")
    memory.add("user", "latest question")
    backend.sessions.save("alice", memory)
    app = create_app(backend, settings=settings(api_key="secret", session_memory_enabled=True))
    headers = {"Authorization": "Bearer secret"}

    info = request(app, "GET", "/v1/sessions/alice/context?reserve_tokens=8", headers=headers)
    assert info.status_code == 200
    payload = info.json()
    assert payload["context_window_tokens"] == 32
    assert payload["categories"]["system_instructions"] > 0
    assert payload["categories"]["tool_definitions"] == 0
    assert payload["non_persisted_categories"] == ["tool_definitions", "files", "tool_results"]
    compacted = request(app, "POST", "/v1/sessions/alice/context/compact?reserve_tokens=8", headers=headers)
    assert compacted.status_code == 200
    assert compacted.json()["after"]["used_tokens"] <= compacted.json()["before"]["used_tokens"]


def test_runtime_selects_token_step_scheduler_for_capable_backend():
    class TokenBackend(FakeBackend):
        async def start_stream(self, request):
            return [request.prompt, 0]

        async def decode_stream_batch(self, states):
            output = []
            for state in states:
                state[1] += 1
                output.append((BackendStreamEvent(
                    token=state[0], completion_tokens=state[1],
                    finish_reason=FinishReason.STOP if state[1] == 1 else None,
                ), state[1] == 1))
            return output

    async def scenario():
        runtime = ServingRuntime(TokenBackend(), continuous_streams=4)
        await runtime.startup()
        events = [event async for event in runtime.stream(GenerateRequest(prompt="batched"))]
        metrics = runtime.metrics()
        await runtime.shutdown()
        return events, metrics

    events, metrics = asyncio.run(scenario())
    assert events[0].token == "batched"
    assert metrics["stream_scheduler_mode"] == "token_step"


def test_runtime_metrics_include_latency_ttft_throughput_and_error_rate():
    async def scenario():
        runtime = ServingRuntime(FakeBackend())
        await runtime.startup()
        await runtime.generate(GenerateRequest(prompt="one"))
        [event async for event in runtime.stream(GenerateRequest(prompt="two"))]
        metrics = runtime.metrics()
        await runtime.shutdown()
        return metrics

    metrics = asyncio.run(scenario())
    assert metrics["latency_p50_seconds"] >= 0
    assert metrics["latency_p95_seconds"] >= metrics["latency_p50_seconds"]
    assert metrics["ttft_p50_seconds"] >= 0
    assert metrics["tokens_per_second"] > 0
    assert metrics["error_rate"] == 0


def test_runtime_reports_paged_kv_utilization_when_backend_exposes_an_allocator():
    class Allocator:
        free_pages = [3, 4]
        tables = {"active": [0, 1, 2]}

    backend = FakeBackend()
    backend.generator = type("Generator", (), {"paged_kv_allocator": Allocator()})()
    metrics = ServingRuntime(backend).metrics()
    assert metrics["paged_kv_pages_used"] == 3
    assert metrics["paged_kv_page_utilization"] == pytest.approx(0.6)


def test_backend_lifecycle_and_readiness():
    async def scenario():
        backend = FakeBackend()
        app = create_app(backend, settings=settings())
        async with app.router.lifespan_context(app):
            assert backend.started
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get("/health/ready")
                assert response.status_code == 200
        assert backend.stopped

    asyncio.run(scenario())


def test_browser_playground_is_served():
    response = request(create_app(FakeBackend(), settings=settings()), "GET", "/ui/")
    assert response.status_code == 200
    assert "Gopi Playground" in response.text
    assert 'src="app.js"' in response.text
    assert 'id="mcpTool"' in response.text
    assert 'id="attachments"' in response.text
    assert 'id="reasoningEffort"' in response.text
    assert 'id="modelSummary"' in response.text
    script = request(create_app(FakeBackend(), settings=settings()), "GET", "/ui/app.js")
    assert "mcp_server:" in script.text
    assert "reasoning_effort:" in script.text
    assert "/resources?context_length=" in script.text


def test_health_reports_whether_the_browser_must_supply_an_api_key():
    open_app = create_app(FakeBackend(), settings=settings())
    protected_app = create_app(FakeBackend(), settings=settings(api_key="secret"))
    assert request(open_app, "GET", "/health/ready").json()["authentication_required"] is False
    assert request(protected_app, "GET", "/health/ready").json()["authentication_required"] is True


def test_swagger_redoc_and_openapi_schema_are_served():
    app = create_app(FakeBackend(), settings=settings())
    swagger = request(app, "GET", "/docs")
    redoc = request(app, "GET", "/redoc")
    schema = request(app, "GET", "/openapi.json")

    assert swagger.status_code == 200
    assert "Swagger UI" in swagger.text
    assert redoc.status_code == 200
    assert "ReDoc" in redoc.text
    assert schema.status_code == 200
    payload = schema.json()
    assert payload["info"]["title"] == "Gopi LLM API"
    assert "/v1/generate" in payload["paths"]
    assert payload["paths"]["/v1/generate"]["post"]["tags"] == ["generation"]
    assert payload["paths"]["/v1/generate"]["post"]["security"]
    assert "HTTPBearer" in payload["components"]["securitySchemes"]


def test_workspace_agent_is_disabled_by_default():
    app = create_app(FakeBackend(), settings=settings())
    response = request(app, "POST", "/v1/workspace/actions", json={
        "actions": [{"type": "read", "path": "README.md"}]
    })
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "workspace_agent_disabled"


def test_authenticated_workspace_agent_reads_files(tmp_path, monkeypatch):
    async def run_immediately(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr("serving.api.run_in_threadpool", run_immediately)
    (tmp_path / "example.txt").write_text("hello workspace\n", encoding="utf-8")
    configured = settings(
        api_key="secret", workspace_agent_enabled=True, workspace_root=str(tmp_path)
    )
    app = create_app(FakeBackend(), settings=configured)
    response = request(
        app, "POST", "/v1/workspace/actions",
        headers={"Authorization": "Bearer secret"},
        json={"actions": [{"type": "read", "path": "example.txt"}]},
    )
    assert response.status_code == 200
    assert response.json()["results"][0]["result"]["content"] == "hello workspace\n"


def test_generate_request_validates_text_attachments():
    request_model = GenerateRequest(prompt="review", attachments=[{
        "name": "example.py", "content": "print('ok')", "media_type": "text/x-code"
    }])
    assert request_model.attachments[0].name == "example.py"
    with pytest.raises(ValueError):
        GenerateRequest(prompt="review", attachments=[{"name": "../secret", "content": "x"}])


def test_production_security_headers_and_trusted_hosts():
    app = create_app(FakeBackend(), settings=settings(allowed_hosts=("safe.example",)))
    blocked = request(app, "GET", "/health/live", headers={"Host": "evil.example"})
    accepted = request(app, "GET", "/health/live", headers={"Host": "safe.example"})

    assert blocked.status_code == 400
    assert accepted.status_code == 200
    assert accepted.headers["x-content-type-options"] == "nosniff"
    assert accepted.headers["x-frame-options"] == "DENY"
    assert accepted.headers["referrer-policy"] == "no-referrer"
    assert "frame-ancestors 'none'" in accepted.headers["content-security-policy"]


def test_docs_can_be_disabled_and_metrics_can_require_authentication():
    app = create_app(
        FakeBackend(),
        settings=settings(api_key="secret", docs_enabled=False, protect_metrics=True),
    )

    assert request(app, "GET", "/docs").status_code == 404
    assert request(app, "GET", "/openapi.json").status_code == 404
    assert request(app, "GET", "/metrics").status_code == 401
    response = request(
        app, "GET", "/metrics", headers={"Authorization": "Bearer secret"}
    )
    assert response.status_code == 200


def test_cors_preflight_allows_bearer_authorization_for_trusted_origin():
    app = create_app(
        FakeBackend(), settings=settings(cors_origins=("https://ui.example",))
    )
    response = request(
        app,
        "OPTIONS",
        "/v1/generate",
        headers={
            "Origin": "https://ui.example",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization,content-type",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "https://ui.example"
    assert "Authorization" in response.headers["access-control-allow-headers"]


def test_rest_generation_response_and_request_id():
    response = request(
        create_app(FakeBackend(), settings=settings()),
        "POST",
        "/v1/generate",
        headers={"X-Request-ID": "request-123"},
        json={"prompt": "hello", "max_tokens": 10, "temperature": 0.2},
    )
    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "request-123"
    payload = response.json()
    assert payload["id"].startswith("gen_")
    assert payload["model"] == "gopi-test"
    assert payload["bot_name"] == "Gopi"
    assert payload["text"] == "Gopi: hello"
    assert payload["finish_reason"] == "stop"
    assert payload["usage"]["total_tokens"] == 5


def test_openai_model_discovery_and_chat_completion():
    class HistoryBackend(FakeBackend):
        async def generate(self, request):
            assert request._chat_messages == [
                {"role": "system", "content": "Answer briefly."},
                {"role": "user", "content": "hello"},
            ]
            return await super().generate(request)

    app = create_app(HistoryBackend(), settings=settings())
    models = request(app, "GET", "/v1/models")
    response = request(
        app,
        "POST",
        "/v1/chat/completions",
        json={
            "model": "gopi-test",
            "messages": [
                {"role": "system", "content": "Answer briefly."},
                {"role": "user", "content": "hello"},
            ],
            "max_tokens": 10,
            "temperature": 0.2,
        },
    )

    assert models.status_code == 200
    assert models.json()["data"][0]["id"] == "gopi-test"
    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "chat.completion"
    assert payload["choices"][0]["message"] == {
        "role": "assistant", "content": "Gopi: hello",
    }
    assert payload["usage"]["total_tokens"] == 5


def test_openai_chat_completion_streams_sse():
    response = request(
        create_app(FakeBackend(), settings=settings()),
        "POST",
        "/v1/chat/completions",
        json={
            "model": "gopi-test",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert '"role": "assistant"' in response.text
    assert '"content": "Go"' in response.text
    assert '"content": "pi"' in response.text
    assert "data: [DONE]" in response.text


def test_openai_chat_completion_rejects_unknown_model():
    response = request(
        create_app(FakeBackend(), settings=settings()),
        "POST",
        "/v1/chat/completions",
        json={
            "model": "not-installed",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_generation_request"


def test_invalid_request_returns_structured_error():
    response = request(
        create_app(FakeBackend(), settings=settings()),
        "POST",
        "/v1/generate",
        json={"prompt": "   ", "unexpected": True},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
    assert response.json()["error"]["request_id"]


def test_context_overflow_returns_client_error() -> None:
    class OverflowBackend(FakeBackend):
        async def generate(self, request):
            raise InvalidGenerationRequestError("prompt exceeds model context")

    response = request(
        create_app(OverflowBackend(), settings=settings()),
        "POST", "/v1/generate", json={"prompt": "too long"},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_generation_request"


def test_serving_settings_reject_invalid_capacity() -> None:
    with pytest.raises(ValueError, match="positive"):
        ServingSettings(max_concurrency=0)


def test_unavailable_backend_returns_503():
    response = request(
        create_app(UnavailableBackend(), settings=settings()),
        "POST",
        "/v1/generate",
        json={"prompt": "hello"},
    )
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "backend_unavailable"


class SlowBackend(FakeBackend):
    async def generate(self, request):
        await asyncio.sleep(0.1)
        return await super().generate(request)


def test_generation_timeout_returns_504():
    response = request(
        create_app(
            SlowBackend(), settings=settings(generation_timeout_seconds=0.01)
        ),
        "POST",
        "/v1/generate",
        json={"prompt": "hello"},
    )
    assert response.status_code == 504
    assert response.json()["error"]["code"] == "generation_timeout"


class BrokenBackend(FakeBackend):
    async def generate(self, request):
        raise RuntimeError("secret backend details")


def test_unexpected_backend_error_is_sanitized():
    response = request(
        create_app(BrokenBackend(), settings=settings()),
        "POST",
        "/v1/generate",
        json={"prompt": "hello"},
    )
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal_error"
    assert "secret" not in response.text


class FakeWebSocket:
    def __init__(self, app, messages):
        self.app = app
        self.headers = {}
        self.messages = list(messages)
        self.sent = []
        self.accepted = False
        self.closed_code = None

    async def accept(self):
        self.accepted = True

    async def close(self, code):
        self.closed_code = code

    async def receive_json(self):
        if not self.messages:
            raise WebSocketDisconnect()
        return self.messages.pop(0)

    async def send_json(self, payload):
        self.sent.append(payload)


def test_websocket_stream_protocol():
    async def scenario():
        app = create_app(FakeBackend(), settings=settings())
        websocket = FakeWebSocket(app, [{"prompt": "hello", "max_tokens": 4}])
        async with app.router.lifespan_context(app):
            await generate_stream(websocket)
        return websocket

    websocket = asyncio.run(scenario())
    assert websocket.accepted
    assert [message["type"] for message in websocket.sent] == [
        "start",
        "token",
        "token",
        "done",
    ]
    assert [websocket.sent[1]["token"], websocket.sent[2]["token"]] == ["Go", "pi"]
    assert websocket.sent[-1]["usage"]["total_tokens"] == 4


def test_websocket_validation_error_does_not_crash_connection():
    async def scenario():
        app = create_app(FakeBackend(), settings=settings())
        websocket = FakeWebSocket(app, [{"prompt": ""}])
        await generate_stream(websocket)
        return websocket

    websocket = asyncio.run(scenario())
    assert websocket.sent[0]["type"] == "error"
    assert websocket.sent[0]["error"]["code"] == "validation_error"


def test_stop_sequences_are_deduplicated():
    request_model = GenerateRequest(prompt="hello", stop=["END", "END", "STOP"])
    assert request_model.stop == ["END", "STOP"]


def test_request_defaults_match_local_inference_profile():
    request_model = GenerateRequest(prompt="hello")
    assert request_model.max_tokens == 128
    assert request_model.temperature == 0.7
    assert request_model.top_k == 40
    assert request_model.top_p == 0.9
    assert request_model.repetition_penalty == 1.1
    assert request_model.response_format is None
    assert request_model.web_search is False
    assert request_model.rag is False


def test_request_accepts_supported_response_formats():
    assert GenerateRequest(prompt="hello", response_format="markdown").response_format == "markdown"
    with pytest.raises(ValueError):
        GenerateRequest(prompt="hello", response_format="html")


def test_request_accepts_supported_chat_modes():
    assert GenerateRequest(prompt="hello").mode == "balanced"
    assert GenerateRequest(prompt="hello", mode="coding").mode == "coding"
    with pytest.raises(ValueError):
        GenerateRequest(prompt="hello", mode="unsupported")


def test_request_accepts_supported_tools():
    assert GenerateRequest(prompt="hello", tools=["calculator", "calculator"]).tools == ["calculator"]
    with pytest.raises(ValueError):
        GenerateRequest(prompt="hello", tools=["shell"])


def test_settings_load_yaml_and_environment_override(tmp_path, monkeypatch):
    config = tmp_path / "inference.yaml"
    config.write_text(
        "bot_name: ConfigGopi\n"
        "serving:\n"
        "  model_name: configured-model\n"
        "  max_concurrency: 7\n"
        "  queue_timeout_seconds: 2.5\n"
        "  generation_timeout_seconds: 30\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("GOPI_INFERENCE_CONFIG", str(config))
    monkeypatch.setenv("GOPI_BOT_NAME", "EnvironmentGopi")
    loaded = ServingSettings.from_environment()
    assert loaded.model_name == "configured-model"
    assert loaded.bot_name == "EnvironmentGopi"
    assert loaded.max_concurrency == 7
    assert loaded.queue_timeout_seconds == 2.5
    assert loaded.generation_timeout_seconds == 30


def test_mcp_can_be_explicitly_disabled_for_minimal_deployment(monkeypatch):
    monkeypatch.setenv("GOPI_MCP_ENABLED", "false")
    assert _load_mcp_config() == {"enabled": False}


def test_invalid_mcp_enabled_override_is_rejected(monkeypatch):
    monkeypatch.setenv("GOPI_MCP_ENABLED", "sometimes")
    with pytest.raises(ValueError, match="boolean"):
        _load_mcp_config()


def test_websocket_rejects_disallowed_browser_origin():
    async def scenario():
        configured = settings(cors_origins=("https://allowed.example",))
        app = create_app(FakeBackend(), settings=configured)
        websocket = FakeWebSocket(app, [])
        websocket.headers["origin"] = "https://blocked.example"
        await generate_stream(websocket)
        return websocket

    websocket = asyncio.run(scenario())
    assert not websocket.accepted
    assert websocket.closed_code == 1008


def test_websocket_accepts_bearer_subprotocol_authentication():
    async def scenario():
        app = create_app(FakeBackend(), settings=settings(api_key="secret"))
        websocket = FakeWebSocket(app, [])
        websocket.headers["sec-websocket-protocol"] = "bearer, secret"
        await generate_stream(websocket)
        return websocket

    websocket = asyncio.run(scenario())
    assert websocket.accepted
    assert websocket.closed_code is None


def test_runtime_rejects_saturated_queue():
    class BlockingBackend(FakeBackend):
        def __init__(self):
            super().__init__()
            self.release = asyncio.Event()

        async def generate(self, request):
            await self.release.wait()
            return await super().generate(request)

    async def scenario():
        backend = BlockingBackend()
        runtime = ServingRuntime(
            backend,
            max_concurrency=1,
            queue_timeout_seconds=0.01,
            generation_timeout_seconds=1,
        )
        first = asyncio.create_task(runtime.generate(GenerateRequest(prompt="hello")))
        while runtime.active_requests == 0:
            await asyncio.sleep(0)
        with pytest.raises(ServerBusyError):
            await runtime.generate(GenerateRequest(prompt="second"))
        backend.release.set()
        await first
        assert runtime.active_requests == 0
        assert runtime.total_requests == 1

    asyncio.run(scenario())


def test_runtime_timeout_releases_worker():
    async def scenario():
        runtime = ServingRuntime(
            SlowBackend(),
            max_concurrency=1,
            queue_timeout_seconds=0.1,
            generation_timeout_seconds=0.01,
        )
        with pytest.raises(GenerationTimeoutError):
            await runtime.generate(GenerateRequest(prompt="hello"))
        assert runtime.active_requests == 0

    asyncio.run(scenario())


def test_websocket_stream_with_session_id(tmp_path):
    from inference.context import SQLiteSessionStore
    from serving.backend import ConfiguredModelBackend
    from tokenizer.bpe import BYTE_ENCODER
    from tokenizer.encoder import DEFAULT_SPECIAL_TOKENS, Tokenizer

    pieces = list(DEFAULT_SPECIAL_TOKENS) + list(BYTE_ENCODER.values())
    vocab = {piece: index for index, piece in enumerate(pieces)}
    tok = Tokenizer(vocab, special_tokens={piece: vocab[piece] for piece in DEFAULT_SPECIAL_TOKENS})

    class SessionFakeBackend(FakeBackend):
        def __init__(self, db_path):
            super().__init__()
            from collections import defaultdict
            self.system_prompt = "You are Gopi."
            self.sessions = SQLiteSessionStore(db_path, tok, max_tokens=512, system_prompt=self.system_prompt)
            self._session_locks = defaultdict(asyncio.Lock)
            self._stream_steps = ConfiguredModelBackend._stream_steps.__get__(self)

        async def stream(self, request):
            async for event in ConfiguredModelBackend.stream(self, request):
                yield event

        async def _stream_unlocked(self, request):
            async for event in ConfiguredModelBackend._stream_unlocked(self, request):
                yield event

        @property
        def generator(self):
            class DummyGenerator:
                tokenizer = tok

                def stream(self, prompt, **options):
                    yield GenerationStep(token="Hi", token_id=1, prompt_tokens=5, completion_tokens=1)
                    yield GenerationStep(token="", token_id=None, prompt_tokens=5, completion_tokens=1, finish_reason="stop")

            from inference.generator import GenerationStep
            return DummyGenerator()

    async def scenario():
        backend = SessionFakeBackend(tmp_path / "sessions.sqlite")
        app = create_app(backend, settings=settings())
        websocket = FakeWebSocket(app, [{"prompt": "hello", "session_id": "sess-123", "max_tokens": 512}])
        async with app.router.lifespan_context(app):
            await generate_stream(websocket)
        return websocket

    websocket = asyncio.run(scenario())
    assert websocket.accepted
    assert [message["type"] for message in websocket.sent] == ["start", "token", "done"], websocket.sent
    assert websocket.sent[1]["token"] == "Hi"


def test_model_resource_planner_endpoint_uses_config_without_loading_weights():
    backend = FakeBackend()
    backend.model_config = "configs/model.hybrid.gpu.yaml"
    app = create_app(backend, settings=settings())
    response = request(
        app, "GET",
        "/v1/models/gopi-test/resources?context_length=4096&batch_size=2&weight_precision=q1_0&kv_precision=int8&memory_gib=4",
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "model.resources"
    assert payload["estimate"]["context_length"] == 4096
    assert payload["estimate"]["weight_precision"] == "q1_0"
    assert payload["estimate"]["kv_precision"] == "int8"
    assert len(payload["alternatives"]) == 12


def test_model_capabilities_expose_hybrid_and_moe_architecture_metadata():
    from runtime.capabilities import discover_capabilities
    config = {
        "vocab_size": 64, "hidden_size": 32, "layers": 8, "heads": 4,
        "kv_heads": 2, "max_position": 1024,
        "attention_layer_pattern": ["linear", "linear", "linear", "dense"],
        "ffn_type": "moe", "num_experts": 8, "experts_per_token": 2,
        "mtp_num_predictions": 2, "qk_norm": True,
    }
    caps = discover_capabilities(FakeBackend(), model_config=config)
    assert caps.attention_pattern == "hybrid"
    assert caps.linear_attention_layers == 6
    assert caps.full_attention_layers == 2
    assert caps.ffn_type == "moe"
    assert caps.num_experts == 8 and caps.experts_per_token == 2
    assert caps.mtp_predictions == 2 and caps.qk_norm is True
