# API reference

Start the local server:

```bash
GOPI_API_KEY='replace-with-a-long-random-key' \
GOPI_CHECKPOINT_PATH=checkpoints/v2-dpo/best.pt \
GOPI_DEVICE=cuda \
.venv/bin/python scripts/serve.py --host 127.0.0.1 --port 8000
```

Interactive OpenAPI documentation is available at `/docs`, ReDoc at `/redoc`,
and the schema at `/openapi.json`.

## Authentication

When `GOPI_API_KEY` is set, all `/v1/*` routes require:

```text
Authorization: Bearer <key>
```

Health endpoints remain public for orchestrator probes. Set
`GOPI_PROTECT_METRICS=true` to require the same bearer key on `/metrics`.
Production Compose enables this setting. Protect or disable documentation at
the application or reverse-proxy layer.

## Health and operations

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/` | Basic liveness response |
| GET | `/health/live` | Process liveness |
| GET | `/health/ready` | Model readiness; returns 503 until ready |
| GET | `/metrics` | Runtime counters and readiness |
| POST | `/v1/admin/reload` | Reload supported backend; requires configured API key |

## Native generation

`POST /v1/generate` accepts:

- required `prompt`;
- `mode`: `balanced`, `creative`, `precise`, or `coding`;
- `max_tokens`, `temperature`, `top_k`, `top_p`, and
  `repetition_penalty`;
- optional `seed`, `stop`, `session_id`, `response_format`, local `tools`,
  `web_search`, `rag`, MCP settings, and `attachments`.

Each attachment has `name`, UTF-8 `content`, and an optional text `media_type`.
The request permits at most eight attachments of 256 KiB each. They are treated
as untrusted reference text, not executable instructions.

Example:

```bash
curl -s http://127.0.0.1:8000/v1/generate \
  -H 'Authorization: Bearer replace-with-a-long-random-key' \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Explain a computer simply.","max_tokens":100,"temperature":0.7}'
```

The response contains generated `text`, `finish_reason`, and prompt/completion
token usage.

## Local-document RAG

Build and enable the persistent retrieval index:

```bash
.venv/bin/pip install -e '.[rag]'  # needed only when ingesting PDF files
.venv/bin/python scripts/build_rag_index.py documents/ --output data/rag/index.sqlite
```

Set `rag.enabled: true` under `configs/inference.yaml`, restart the server,
and call the native endpoint with `"rag": true`. Alternatively prefix a prompt
with `/rag`; this also works through OpenAI-compatible chat clients. The server
uses disk-backed SQLite FTS5/BM25 retrieval, does not send documents to an external service,
limits context per chunk, labels retrieved content as untrusted, and appends
`document://...#chunk-N` citations. Set both `rag` and `web_search` to `true`,
or prefix the question with `/hybrid`, to combine local and current web
sources. Supported local formats are TXT, Markdown, HTML, CSV/TSV, JSON/JSONL,
YAML, and PDF.

## OpenAI-compatible routes

`GET /v1/models` lists the configured local model. `POST
/v1/chat/completions` supports the text-only subset used by common chat UIs.

Supported request fields are `model`, `messages`, `stream`, `max_tokens`,
`max_completion_tokens`, `temperature`, `top_p`, `stop`, `seed`, and `user`.
Messages support `system`, `user`, and `assistant`. At least one user message is
required. Unknown extra request fields are ignored for UI compatibility.

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Authorization: Bearer replace-with-a-long-random-key' \
  -H 'Content-Type: application/json' \
  -d '{"model":"gopi","messages":[{"role":"user","content":"Hello"}],"max_tokens":64}'
```

The model value must exactly match `GOPI_MODEL_NAME`. This compatibility layer
does not provide embeddings, image/audio input, function calling, tool-call
objects, or the OpenAI Responses API.

## Streaming

Set `"stream": true` on `/v1/chat/completions`. The response uses Server-Sent
Events with `chat.completion.chunk` objects and terminates with:

```text
data: [DONE]
```

The native WebSocket interface is documented by the running OpenAPI-adjacent
usage guide; see [V2_USAGE_GUIDE.md](V2_USAGE_GUIDE.md).

## Workspace agent

The optional `POST /v1/workspace/actions` endpoint executes up to eight ordered,
bounded actions. Set `GOPI_WORKSPACE_AGENT_ENABLED=true`, configure an explicit
`GOPI_WORKSPACE_ROOT`, and set `GOPI_API_KEY` before starting the server.

```bash
curl -s http://127.0.0.1:8000/v1/workspace/actions \
  -H 'Authorization: Bearer replace-with-a-long-random-key' \
  -H 'Content-Type: application/json' \
  -d '{"actions":[{"type":"search","query":"class Generator"},{"type":"read","path":"src/inference/generator.py"}]}'
```

Supported actions are `read`, `search`, `edit`, `patch`, `test`, and `git`.
Edits and patches default to `apply: false`, returning a reviewable preview.
Overwriting an existing file additionally requires the SHA-256 returned by a
fresh `read` action. Tests are restricted to `unit` and `all`; Git is restricted
to `status`, `diff`, `diff_staged`, and `log`. Absolute paths, path traversal,
arbitrary commands, deletion patches, staging, and commits are not exposed.

## Errors and request IDs

Errors use an object containing `code`, `message`, and optional `request_id`.
The server returns an `X-Request-ID` response header and accepts a safe
`X-Request-ID` supplied by the caller. Common statuses are 401 authentication,
422 validation, 429 busy or rate limited, 503 unavailable, and 504 timeout.

## Main environment variables

| Variable | Purpose |
| --- | --- |
| `GOPI_CHECKPOINT_PATH` | Model checkpoint |
| `GOPI_MODEL_CONFIG` | Model YAML path |
| `GOPI_TOKENIZER_PATH` | Tokenizer directory |
| `GOPI_INFERENCE_CONFIG` | Inference/serving YAML path |
| `GOPI_DEVICE` | `cpu`, `cuda`, or supported device |
| `GOPI_MODEL_NAME` | API-visible model ID |
| `GOPI_BOT_NAME` | Assistant display name |
| `GOPI_RAG_INDEX` | Local persistent RAG index path |
| `GOPI_API_KEY` | Optional bearer secret |
| `GOPI_CORS_ORIGINS` | Comma-separated trusted browser origins |
| `GOPI_ALLOWED_HOSTS` | Comma-separated accepted HTTP Host names |
| `GOPI_DOCS_ENABLED` | Enable Swagger, ReDoc, and OpenAPI routes |
| `GOPI_MCP_ENABLED` | Explicitly disable or enable configured MCP servers |
| `GOPI_PROTECT_METRICS` | Require bearer authentication for `/metrics` |
| `GOPI_FORWARDED_ALLOW_IPS` | Trusted reverse-proxy addresses for forwarded headers |
| `GOPI_MAX_CONCURRENCY` | Maximum active generations |
| `GOPI_QUEUE_TIMEOUT_SECONDS` | Queue wait limit |
| `GOPI_GENERATION_TIMEOUT_SECONDS` | Generation time limit |
| `GOPI_REQUESTS_PER_MINUTE` | Per-client rate limit; zero disables it |
| `GOPI_RATE_LIMIT_STORE` | Optional SQLite rate-limit state path |
| `GOPI_WORKSPACE_AGENT_ENABLED` | Enable authenticated workspace actions; disabled by default |
| `GOPI_WORKSPACE_ROOT` | Filesystem root available to workspace actions |

Treat configuration behavior as versioned API behavior and inspect `/docs` for
the exact schema of the currently running build.
