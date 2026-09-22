import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from inference.context import SQLiteSessionStore
from serving.api import ResponsesRequest
from serving.schemas import GenerateRequest, OpenAIChatCompletionRequest
from tokenizer.bpe import BYTE_ENCODER
from tokenizer.encoder import DEFAULT_SPECIAL_TOKENS, Tokenizer


ROOT = Path(__file__).resolve().parents[2]


def test_ui_contains_every_protected_backend_feature_route():
    script = (ROOT / "ui" / "app.js").read_text(encoding="utf-8")
    required = [
        "/v1/generate", "/v1/generate/stream", "/v1/chat/completions", "/v1/responses",
        "/v1/models", "/v1/embeddings", "/v1/embeddings/models", "/v1/audit/events",
        "/v1/sessions", "/v1/sessions/", "/v1/workspace/actions", "/v1/requests/", "/admin/models",
        "/context?reserve_tokens=", "/context/compact?reserve_tokens=", "/resources?context_length=",
    ]
    missing = [route for route in required if route not in script]
    assert not missing, f"UI is missing backend routes: {missing}"


def test_ui_does_not_store_secret_credentials():
    script = (ROOT / "ui" / "app.js").read_text(encoding="utf-8")
    settings_block = script.split("function saveSettings", 1)[1].split("function loadSettings", 1)[0]
    assert "apiKey" not in settings_block
    assert "adminApiKey" not in settings_block


def test_ui_defers_protected_requests_until_an_api_key_is_entered():
    script = (ROOT / "ui" / "app.js").read_text(encoding="utf-8")
    assert "authenticationRequired=!!body.authentication_required" in script
    assert "This server requires an API key." in script
    assert 'el.apiKey.addEventListener("change",checkHealth)' in script
    assert "function adminHeaders" in script


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_ui_javascript_compiles():
    result = subprocess.run(["node", "--check", "ui/app.js"], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_generate_schema_accepts_ui_date_time_tool():
    payload = GenerateRequest.model_validate({"prompt": "what time is it?", "tools": ["datetime", "datetime"]})
    assert payload.tools == ["datetime"]


def test_chat_extensions_match_ui_controls():
    payload = OpenAIChatCompletionRequest.model_validate({
        "model": "gopi",
        "messages": [{"role": "user", "content": "hello"}],
        "session_id": "chat-abc_123",
        "mode": "coding",
        "web_search": True,
        "rag": True,
        "mcp": True,
        "mcp_server": "filesystem",
        "no_repeat_ngram_size": 4,
        "min_tokens": 3,
    })
    request = payload.generation_request("gopi")
    assert request.session_id == "chat-abc_123"
    assert request.mode == "coding"
    assert request.web_search and request.rag and request.mcp
    assert request.mcp_server == "filesystem"
    assert request.no_repeat_ngram_size == 4
    assert request.min_tokens == 3


def test_responses_schema_rejects_bad_ui_input_and_accepts_features():
    with pytest.raises(ValueError, match="input cannot be empty"):
        ResponsesRequest.model_validate({"model": "gopi", "input": " "})
    request = ResponsesRequest.model_validate({
        "model": "gopi", "input": "hello", "mode": "precise", "mcp": True,
        "mcp_server": "filesystem", "repetition_penalty": 1.2, "no_repeat_ngram_size": 4,
        "min_tokens": 2,
    })
    assert request.mode == "precise"
    assert request.mcp and request.mcp_server == "filesystem"


def _tokenizer():
    pieces = list(DEFAULT_SPECIAL_TOKENS) + list(BYTE_ENCODER.values())
    vocab = {piece: index for index, piece in enumerate(pieces)}
    return Tokenizer(vocab, special_tokens={p: vocab[p] for p in DEFAULT_SPECIAL_TOKENS})


def test_session_store_review_approve_export_delete_lifecycle(tmp_path):
    store = SQLiteSessionStore(tmp_path / "sessions.sqlite", _tokenizer(), max_tokens=128, system_prompt="")
    memory = store.load("chat-test")
    memory.add("user", "What is two plus two?")
    memory.add("assistant", "4")
    store.save("chat-test", memory)

    listed = store.list_sessions()
    assert listed and listed[0]["session_id"] == "chat-test"
    review = store.review_last("chat-test")
    assert review == {"prompt": "What is two plus two?", "answer": "4"}
    assert store.approve_last("chat-test", corrected_response="Four") == 1
    exported = store.export_training_jsonl("chat-test")
    assert json.loads(exported)["messages"][-1]["content"] == "Four"
    assert store.delete_training("chat-test") == 1
    with pytest.raises(ValueError, match="no approved"):
        store.export_training_jsonl("chat-test")
    store.delete("chat-test", include_training_examples=True)
    assert store.load("chat-test").snapshot() == ()


def test_session_history_http_contract_and_training_controls(tmp_path):
    import asyncio
    from tests.test_serving import FakeBackend, request, settings
    backend = FakeBackend()
    backend.sessions = SQLiteSessionStore(tmp_path / "sessions.sqlite", _tokenizer(), max_tokens=128, system_prompt="")
    memory = backend.sessions.load("chat-test")
    memory.add("user", "Tell me a safe fact")
    memory.add("assistant", "Water freezes at 0 C under standard pressure.")
    backend.sessions.save("chat-test", memory)
    app = __import__("serving.api", fromlist=["create_app"]).create_app(
        backend, settings=settings(api_key="secret", session_memory_enabled=True)
    )
    headers = {"Authorization": "Bearer secret"}

    assert request(app, "GET", "/v1/sessions", headers=headers).status_code == 200
    review = request(app, "GET", "/v1/sessions/chat-test/training/review", headers=headers)
    assert review.status_code == 200
    assert review.json()["prompt"] == "Tell me a safe fact"

    approved = request(
        app, "POST", "/v1/sessions/chat-test/training/approve", headers=headers,
        json={"approved": True, "corrected_response": "Corrected safe fact."},
    )
    assert approved.status_code == 200 and approved.json()["example_count"] == 1

    exported = request(app, "GET", "/v1/sessions/chat-test/training/export", headers=headers)
    assert exported.status_code == 200
    assert "Corrected safe fact." in exported.text

    deleted_training = request(app, "DELETE", "/v1/sessions/chat-test/training", headers=headers)
    assert deleted_training.status_code == 200 and deleted_training.json()["deleted_count"] == 1
    deleted = request(app, "DELETE", "/v1/sessions/chat-test/memory", headers=headers)
    assert deleted.status_code == 200 and deleted.json()["deleted"] is True
    assert request(app, "GET", "/v1/sessions/chat-test/memory", headers=headers).json()["messages"] == []


def test_admin_key_can_be_separate_from_normal_api_key():
    from tests.test_serving import FakeBackend, request, settings
    from serving.api import create_app
    app = create_app(
        FakeBackend(), settings=settings(api_key="normal-key", admin_api_key="admin-key")
    )
    normal = request(app, "GET", "/admin/models", headers={"Authorization": "Bearer normal-key"})
    separate = request(
        app, "GET", "/admin/models",
        headers={"Authorization": "Bearer normal-key", "X-Admin-API-Key": "admin-key"},
    )
    wrong = request(
        app, "GET", "/admin/models",
        headers={"Authorization": "Bearer normal-key", "X-Admin-API-Key": "wrong"},
    )
    assert normal.status_code == 403
    assert separate.status_code == 200
    assert wrong.status_code == 403


def test_invalid_http_generation_request_returns_human_readable_field_error():
    from tests.test_serving import FakeBackend, request, settings
    from serving.api import create_app
    response = request(
        create_app(FakeBackend(), settings=settings()), "POST", "/v1/generate",
        json={"prompt": "hello", "top_p": 2},
    )
    assert response.status_code == 422
    message = response.json()["error"]["message"]
    assert message.startswith("Invalid")
    assert "top_p" in message


def test_session_memory_without_api_key_is_configuration_error_not_forbidden():
    from tests.test_serving import FakeBackend, request, settings
    from serving.api import create_app
    app = create_app(FakeBackend(), settings=settings(api_key=None, session_memory_enabled=True))
    response = request(app, "DELETE", "/v1/sessions/chat-test/training")
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "session_memory_auth_not_configured"
    assert "GOPI_API_KEY" in response.json()["error"]["message"]


def test_cors_allows_ui_delete_operations():
    from tests.test_serving import FakeBackend, settings
    from serving.api import create_app
    app = create_app(
        FakeBackend(),
        settings=settings(
            api_key="secret",
            cors_origins=("http://localhost:3000",),
            session_memory_enabled=True,
        ),
    )
    from tests.asgi_client import ASGIClient
    with ASGIClient(app) as client:
        response = client.options(
            "/v1/sessions/chat-test/training",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "DELETE",
                "Access-Control-Request-Headers": "authorization,content-type",
            },
        )
        assert response.status_code == 200
        assert "DELETE" in response.headers.get("access-control-allow-methods", "")
