# Production Serving & API Protocols (`docs/SERVING.md`)

*Authoritative Source: [`src/serving/api.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/serving/api.py), [`src/serving/websocket.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/serving/websocket.py), [`src/serving/batching.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/serving/batching.py), [`scripts/serve.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/scripts/serve.py)*

---

## 1. Serving Architecture

The serving subsystem provides high-throughput concurrent inference using FastAPI, Uvicorn, and an asynchronous batching scheduler:

```text
Client Request
      │
      ├── HTTP POST /v1/chat/completions ──┐
      └── WebSocket /ws/generate ──────────┼──> Rate Limiter (src/serving/rate_limit.py)
                                           │
                                           ▼
                                Dynamic Batch Queue (src/serving/batching.py)
                                           │
                                           ▼
                                Inference Runtime & Paged KV Cache
                                           │
                        ┌──────────────────┴──────────────────┐
                        ▼                                     ▼
                HTTP JSON Response                    WebSocket Token Stream
```

---

## 2. API Endpoints

### 2.1 HTTP Endpoints

| Method | Path | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Liveness and model readiness probe |
| `GET` | `/metrics` | Prometheus-style metrics (TTFT, TPS, KV cache occupancy) |
| `GET` | `/v1/models` | List available loaded models |
| `POST`| `/v1/chat/completions` | OpenAI-compatible chat completion endpoint |
| `POST`| `/v1/completions` | Raw prompt text completion endpoint |

### 2.2 WebSocket Streaming (`/ws/generate`)
Full-duplex WebSocket connection for real-time token streaming.

**Client Request Message**:
```json
{
  "prompt": "Explain quantum computing simply.",
  "max_tokens": 150,
  "temperature": 0.7,
  "top_p": 0.9
}
```

**Server Stream Event**:
```json
{
  "event": "token",
  "token": "Quantum",
  "token_id": 18241,
  "index": 0
}
```

**Server Complete Event**:
```json
{
  "event": "done",
  "finish_reason": "stop",
  "usage": {
    "prompt_tokens": 6,
    "completion_tokens": 48,
    "total_tokens": 54
  }
}
```

---

## 3. Dynamic Batching & Concurrency

The dynamic batcher in [`src/serving/batching.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/serving/batching.py) aggregates incoming requests up to `max_batch_size: 4` with a maximum wait timeout of `batch_timeout_ms: 10`. On 4 GB GPUs, this maximizes compute utilization while preventing memory exhaustion.

---

## 4. Starting the Server

```bash
# Start server with default inference configuration
.venv/bin/python scripts/serve.py \
  --config configs/inference.yaml \
  --host 0.0.0.0 \
  --port 8000
```
Web client is accessible at `http://localhost:8000/ui/`.

