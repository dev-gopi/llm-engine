# V2 model usage guide

This guide explains how to use a trained v2 model from the command line, an
interactive terminal chat, the browser UI, the REST API, and an exported
SafeTensors bundle. Run commands from the repository root.

For training from datasets through DPO, read
[V2_TRAINING_GUIDE.md](V2_TRAINING_GUIDE.md).

## 1. Choose the checkpoint

Use the most advanced validation-selected checkpoint that exists:

```text
1. checkpoints/v2-dpo/best.pt          final preference-aligned assistant
2. checkpoints/v2-finetuning/best.pt   supervised chat/instruction model
3. checkpoints/v2-pretraining/best.pt  raw text-completion model
```

Use `best.pt` for generation. `latest.pt` exists primarily to resume interrupted
training and may not be the best model.

The examples below use the DPO checkpoint. Replace it with the fine-tuning
checkpoint when DPO was skipped. Use the pretraining checkpoint only for raw
completion tests.

## 2. Verify files and hardware

```bash
.venv/bin/python scripts/capabilities.py
```

Check the final checkpoint:

```bash
test -f checkpoints/v2-dpo/best.pt && echo "Final checkpoint found"
```

Required runtime files are:

```text
configs/model.gpu.yaml
configs/inference.yaml
data/tokenizer-v2/
checkpoints/v2-dpo/best.pt
```

## 3. Generate one answer from the CLI

English assistant response:

```bash
.venv/bin/python scripts/generate.py \
  "Explain artificial intelligence in simple language." \
  --model-config configs/model.gpu.yaml \
  --inference-config configs/inference.yaml \
  --tokenizer data/tokenizer-v2 \
  --checkpoint checkpoints/v2-dpo/best.pt \
  --device cuda \
  --max-tokens 150 \
  --temperature 0.7
```

Bengali:

```bash
.venv/bin/python scripts/generate.py \
  "বাংলায় কম্পিউটার কী তা সহজভাবে ব্যাখ্যা করো।" \
  --model-config configs/model.gpu.yaml \
  --inference-config configs/inference.yaml \
  --tokenizer data/tokenizer-v2 \
  --checkpoint checkpoints/v2-dpo/best.pt \
  --device cuda \
  --max-tokens 150 \
  --temperature 0.7
```

Hindi:

```bash
.venv/bin/python scripts/generate.py \
  "कंप्यूटर क्या है? सरल भाषा में समझाइए।" \
  --model-config configs/model.gpu.yaml \
  --inference-config configs/inference.yaml \
  --tokenizer data/tokenizer-v2 \
  --checkpoint checkpoints/v2-dpo/best.pt \
  --device cuda \
  --max-tokens 150 \
  --temperature 0.7
```

Coding:

```bash
.venv/bin/python scripts/generate.py \
  "Write a Python function that returns the factorial of a non-negative integer." \
  --model-config configs/model.gpu.yaml \
  --inference-config configs/inference.yaml \
  --tokenizer data/tokenizer-v2 \
  --checkpoint checkpoints/v2-dpo/best.pt \
  --device cuda \
  --max-tokens 200 \
  --temperature 0.2 \
  --response-format markdown
```

Lower temperature is more deterministic. A value around `0.2` is useful for
code and factual answers; `0.7` is a balanced default; higher values are more
creative but less reliable.

## 4. Test a pretraining checkpoint

Pretraining learns text continuation rather than assistant behavior. Use
`--raw` so the prompt is not wrapped in the chat template:

```bash
.venv/bin/python scripts/generate.py \
  "Once upon a time" \
  --model-config configs/model.gpu.yaml \
  --inference-config configs/inference.yaml \
  --tokenizer data/tokenizer-v2 \
  --checkpoint checkpoints/v2-pretraining/best.pt \
  --device cuda \
  --raw \
  --max-tokens 120 \
  --temperature 0.7
```

Do not judge chat quality from a pretraining checkpoint.

## 5. Start an interactive terminal chat

```bash
.venv/bin/python scripts/chat.py \
  --model-config configs/model.gpu.yaml \
  --inference-config configs/inference.yaml \
  --tokenizer data/tokenizer-v2 \
  --checkpoint checkpoints/v2-dpo/best.pt \
  --device cuda \
  --max-tokens 150 \
  --temperature 0.7
```

Interactive commands:

```text
/help                 list commands
/user <message>       send an explicit user message
/system <prompt>      replace the system prompt
/format markdown      return Markdown-oriented responses
/format plain         return plain text
/search <query>       search the configured provider, then answer
/clear                clear conversation history
/quit                 exit
```

Web search requires a reachable SearXNG endpoint or a configured Brave Search
API key. Normal local chat does not require web search.

## 6. Export the final model

```bash
.venv/bin/python scripts/export.py \
  --model-config configs/model.gpu.yaml \
  --tokenizer data/tokenizer-v2 \
  --checkpoint checkpoints/v2-dpo/best.pt \
  --format safetensors \
  --output exports/v2-final/gopi-v2.safetensors
```

The output bundle contains:

```text
exports/v2-final/gopi-v2.safetensors
exports/v2-final/model.yaml
exports/v2-final/tokenizer/
```

Generate from that bundle without the training checkpoint:

```bash
.venv/bin/python scripts/generate_exported.py \
  "Hello! What can you do?" \
  --model exports/v2-final/gopi-v2.safetensors \
  --model-config exports/v2-final/model.yaml \
  --tokenizer exports/v2-final/tokenizer \
  --device cuda \
  --max-tokens 150 \
  --temperature 0.7 \
  --top-k 40 \
  --top-p 0.9
```

## 7. Start the API server

The server reads `configs/inference.yaml`. Override its checkpoint explicitly
so it serves the final DPO model rather than the pretraining default:

```bash
GOPI_INFERENCE_CONFIG=configs/inference.yaml \
GOPI_MODEL_CONFIG=configs/model.gpu.yaml \
GOPI_TOKENIZER_PATH=data/tokenizer-v2 \
GOPI_CHECKPOINT_PATH=checkpoints/v2-dpo/best.pt \
GOPI_DEVICE=cuda \
.venv/bin/python scripts/serve.py \
  --host 127.0.0.1 \
  --port 8000
```

Keep this terminal open while using the UI or API. Test liveness and model
readiness from another terminal:

```bash
curl http://127.0.0.1:8000/health/live
curl http://127.0.0.1:8000/health/ready
```

`ready: true` means the model loaded successfully. A `503` readiness response
usually means that a checkpoint, tokenizer, model config, device, or memory
requirement is wrong.

## 8. Use the browser UI

Start the server as shown above, then open:

```text
http://127.0.0.1:8000/ui/
```

The built-in UI supports:

- normal and streaming generation;
- balanced, creative, precise, and coding modes;
- plain or Markdown responses;
- temperature, top-k, top-p, repetition penalty, and maximum-token controls;
- conversation sessions and token usage;
- calculator and date/time tools;
- optional web search and allowlisted MCP tools.

Use `127.0.0.1` while running locally. Binding the server to `0.0.0.0` exposes
it to other machines on reachable networks and should be combined with an API
key, firewall, and trusted deployment gateway.

## 9. Use Swagger API documentation

Start the API server, then open the interactive Swagger UI:

```text
http://127.0.0.1:8000/docs
```

Swagger lists health, generation, and operations endpoints. Expand
`POST /v1/generate`, select **Try it out**, edit the JSON request, and select
**Execute**. The model must report ready before generation succeeds.

Alternative API documentation and the raw OpenAPI schema are available at:

```text
http://127.0.0.1:8000/redoc
http://127.0.0.1:8000/openapi.json
```

When `GOPI_API_KEY` is enabled, select **Authorize** in Swagger and enter the
token. Swagger sends it as an `Authorization: Bearer <key>` header to `/v1/*`.
Avoid entering a production secret into a Swagger page served from an untrusted
machine or origin.

## 10. Call the REST API

Balanced response:

```bash
curl -X POST http://127.0.0.1:8000/v1/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "Explain what a computer is.",
    "mode": "balanced",
    "response_format": "plain",
    "max_tokens": 150,
    "temperature": 0.7,
    "top_k": 40,
    "top_p": 0.9,
    "repetition_penalty": 1.1,
    "seed": 42
  }'
```

Coding response:

```bash
curl -X POST http://127.0.0.1:8000/v1/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "Write a Python function that checks whether a number is prime.",
    "mode": "coding",
    "response_format": "markdown",
    "max_tokens": 220,
    "temperature": 0.2
  }'
```

Built-in calculator tool:

```bash
curl -X POST http://127.0.0.1:8000/v1/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "/calc (125 * 8) + 17",
    "tools": ["calculator"],
    "max_tokens": 80
  }'
```

Reuse a validated `session_id` to retain server-side conversation context:

```json
{
  "prompt": "What did I ask before?",
  "session_id": "my-session-1"
}
```

## 11. Enable API-key authentication

Choose a strong secret and pass it through the environment rather than storing
it in Git:

```bash
GOPI_API_KEY='replace-with-a-long-random-secret' \
GOPI_CHECKPOINT_PATH=checkpoints/v2-dpo/best.pt \
GOPI_DEVICE=cuda \
.venv/bin/python scripts/serve.py --host 127.0.0.1 --port 8000
```

Authenticated request:

```bash
curl -X POST http://127.0.0.1:8000/v1/generate \
  -H 'Authorization: Bearer replace-with-a-long-random-secret' \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Hello Gopi","max_tokens":80}'
```

Do not expose an unauthenticated development server directly to the public
internet. Use TLS and a trusted reverse proxy or ingress for deployment.

## 12. Streaming generation

The browser UI uses:

```text
ws://127.0.0.1:8000/v1/generate/stream
```

After connecting, send the same JSON object accepted by the REST endpoint. The
server returns a `start` event, multiple `token` events, and a final `done`
event. With authentication, WebSocket clients must supply an Authorization
header or the supported `bearer, <key>` WebSocket subprotocol. The built-in UI
is the easiest local streaming client.

## 13. Load-test the local API

Start the server first, then run a small laptop-safe test:

```bash
.venv/bin/python scripts/load_test.py \
  --url http://127.0.0.1:8000 \
  --requests 10 \
  --concurrency 1 \
  --prompt-words 16,32 \
  --max-tokens 32 \
  --timeout 180 \
  --max-p95 180 \
  --max-failure-rate 0.01 \
  --output exports/v2-final/load-test.json
```

For an authenticated server, add:

```text
--api-key replace-with-a-long-random-secret
```

Increase concurrency only after the single-request path is stable and GPU
memory has been measured.

## 14. Common problems

### Checkpoint tokenizer fingerprint mismatch

The checkpoint was trained with a different tokenizer. Use its original
tokenizer. Never rebuild the tokenizer vocabulary for an existing checkpoint.

### CUDA out of memory

Stop other GPU processes, shorten prompts and `max_tokens`, reduce serving
concurrency, and use only one model-serving worker on a single GPU.

### UI says the model is not ready

Check the server terminal and verify `GOPI_CHECKPOINT_PATH`,
`GOPI_MODEL_CONFIG`, `GOPI_TOKENIZER_PATH`, and `GOPI_DEVICE`. Then call
`/health/ready` again.

### Output repeats words or phrases

Keep `repetition_penalty` near `1.1`, reduce temperature, use top-p around
`0.9`, and test the validation-selected SFT/DPO checkpoint instead of
`latest.pt` or a raw pretraining checkpoint.

### Bengali, Hindi, reasoning, or coding quality is weak

Generation settings cannot replace missing training quality. Compare domains
with the evaluation commands in the training guide and improve the relevant
SFT data before repeating DPO.

## 15. Connect a third-party GitHub UI

The server implements the OpenAI Chat Completions compatibility endpoints used
by general-purpose clients such as
[Open WebUI](https://github.com/open-webui/open-webui):

```text
GET  /v1/models
POST /v1/chat/completions
```

Both normal JSON responses and `text/event-stream` streaming responses are
supported. Conversation histories containing `system`, `user`, and `assistant`
messages are converted to this model's chat template. Oversized UI histories
are trimmed from the oldest turn to fit the model's 512-token context.

Start Gopi with an API key:

```bash
GOPI_API_KEY='local-gopi-key' \
GOPI_CHECKPOINT_PATH=checkpoints/v2-dpo/best.pt \
GOPI_DEVICE=cuda \
.venv/bin/python scripts/serve.py --host 0.0.0.0 --port 8000
```

In an OpenAI-compatible UI, configure:

```text
API base URL: http://HOST_RUNNING_GOPI:8000/v1
API key:      local-gopi-key
Model:        gopi-v2
```

If the UI runs in Docker on the same Linux machine, `127.0.0.1` inside its
container points to the container itself, not the host. Use an explicit host
address or configure Docker's host-gateway mapping. Keep port 8000 private to a
trusted network and use TLS/reverse-proxy protection before remote exposure.

OpenAI-compatible curl smoke test:

```bash
curl http://127.0.0.1:8000/v1/models \
  -H 'Authorization: Bearer local-gopi-key'

curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Authorization: Bearer local-gopi-key' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gopi-v2",
    "messages": [
      {"role": "system", "content": "Answer clearly and briefly."},
      {"role": "user", "content": "What is a computer?"}
    ],
    "stream": false,
    "max_tokens": 120,
    "temperature": 0.7,
    "top_p": 0.9
  }'
```

Supported compatibility fields include `model`, `messages`, `stream`,
`max_tokens`, `max_completion_tokens`, `temperature`, `top_p`, `stop`, and
`seed`. Unknown extra request fields are ignored for UI interoperability.

The current compatibility layer is text-only. It does not claim OpenAI
embeddings, image generation, speech, transcription, multimodal message parts,
or function-calling compatibility. Use Gopi's native `/v1/generate` endpoint
for its calculator, datetime, web-search, and MCP options.
