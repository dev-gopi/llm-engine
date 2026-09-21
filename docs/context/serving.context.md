# Compact Context: Serving Subsystem (`docs/context/serving.context.md`)

> **AGENT CONTEXT PACK**: Load this file when working on HTTP endpoints, WebSockets, dynamic batching, rate limiting, or server configuration.

---

## 1. Authoritative Sources of Truth
- **FastAPI Endpoints**: [`src/serving/api.py`](../../src/serving/api.py)
- **WebSocket Streaming**: [`src/serving/websocket.py`](../../src/serving/websocket.py)
- **Dynamic Batching**: [`src/serving/batching.py`](../../src/serving/batching.py) (`DynamicBatcher`)
- **Rate Limiting**: [`src/serving/rate_limit.py`](../../src/serving/rate_limit.py)
- **Schemas**: [`src/serving/schemas.py`](../../src/serving/schemas.py)
- **Server Entry**: [`scripts/serve.py`](../../scripts/serve.py)

---

## 2. API Contract
- `POST /v1/chat/completions`: Accepts `{model, messages, temperature, max_tokens, stream}`. Returns OpenAI-compatible chunk or complete object.
- `GET /health`: Returns `{"status": "ok", "model": "loaded"}`.
- `GET /metrics`: Returns active queue length, TTFT, and tokens per second.
- `WS /ws/generate`: Sends `{prompt, max_tokens, ...}`; receives `{event: "token", token: "..."}`.

---

## 3. Key Invariants
1. WebSocket disconnects must release associated KV cache allocations immediately.
2. Dynamic batcher queues must enforce `batch_timeout_ms` to avoid starvation on low traffic.
3. Requests exceeding rate limits return HTTP 429 Too Many Requests with `Retry-After`.

---

## 4. Primary Verification Tests
```bash
.venv/bin/pytest tests/test_serving.py tests/test_serving_orchestration.py -q
```

