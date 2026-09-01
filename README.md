# Gopi LLM Engine

Gopi is an educational, configuration-driven project for building a GPT-style
language model from the ground up. The repository separates tokenization, data
processing, model architecture, training, inference, and serving so each part
can evolve independently as the model grows.

> **Development status:** The local from-scratch pipeline is implemented and
> covered by an automated test suite: preparation, tokenization, pretraining, SFT, DPO
> components, evaluation, cached generation, serving, and export. FSDP,
> distributed checkpoints, binary token shards, and scalable serving primitives
> are opt-in. Multi-GPU/multi-node operation still requires validation on the
> target cluster; this repository is not presented as a production deployment.

## Goals

- Learn and implement the complete LLM pipeline rather than wrapping a hosted model.
- Keep experiments reproducible through YAML configuration.
- Maintain clear boundaries between training and inference code.
- Scale from a small local model toward distributed training and optimized serving.
- Support checkpoint recovery and multiple deployment formats.

## Quick start

New users should follow these end-to-end guides:

- [V2 training guide](docs/V2_TRAINING_GUIDE.md) — collect datasets, prepare the
  tokenizer, pretrain, fine-tune, run DPO, evaluate, and export.
- [V2 usage guide](docs/V2_USAGE_GUIDE.md) — generate from the CLI, chat in the
  terminal, use the browser UI and API, stream responses, and run exported
  models.
- [V2 capabilities and scaling guide](docs/V2_CAPABILITIES_AND_SCALING.md) —
  understand realistic uses, limitations, model sizes, and the path from 54.4M
  toward the provided 1B, 7B, and 30B architecture targets.
- [Troubleshooting guide](docs/TROUBLESHOOTING.md) — diagnose CUDA memory,
  non-finite gradients, tokenizer/checkpoint, validation, and serving failures.
- [Dataset formats](docs/DATASET_FORMATS.md) — required JSONL schemas for
  pretraining, SFT, DPO, and domain evaluation.
- [API reference](docs/API_REFERENCE.md) — native and OpenAI-compatible HTTP
  endpoints, authentication, request fields, streaming, and errors.
- [Deployment guide](docs/DEPLOYMENT.md) — run the local service safely behind
  authentication, TLS, a reverse proxy, and an optional third-party UI.
- [Changelog](CHANGELOG.md) — user-visible project changes by release.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --editable '.[dev]'
python scripts/capabilities.py
pytest -q
```

Inspect a model before allocating its weights:

```bash
python scripts/inspect_model.py configs/model.v2.gpu.yaml
```

Estimate a complete training run without allocating the model:

```bash
python scripts/plan_training.py \
  --model-config configs/model.v2.gpu.yaml \
  --training-config configs/pretraining.v2.gpu.yaml \
  --training-tokens 1000000000 \
  --hardware-tflops 10 --utilization 0.35 --gpu-memory-gib 4 --require-fit
```

CI can also enforce `--max-hours` and `--max-cost`; constraint failures return
exit status 2 while retaining the JSON report.

The v2 GPU profile is the recommended from-scratch laptop path. The 1B/7B/30B
files are architecture targets, not profiles for a 4 GB GPU.

Production-style local HTTPS serving is available through Docker Compose and
the included Nginx reverse proxy:

```bash
cp .env.production.example .env.production
# Replace GOPI_API_KEY in .env.production before continuing.
./deploy/generate_dev_certs.sh
docker compose --env-file .env.production up --build -d
curl --insecure https://localhost:8443/health/ready
```

The generated certificate is for local development only. See the
[deployment guide](docs/DEPLOYMENT.md) for GPU serving, trusted TLS
certificates, authentication, proxy, and security requirements.

## Repository structure

```text
llm-engine/
├── configs/                 # Model, tokenizer, training, and inference settings
├── data/
│   ├── raw/                 # Immutable source datasets
│   ├── processed/           # Normalized training and evaluation data
│   ├── tokenizer/           # Vocabulary and merge artifacts
│   └── cache/               # Reusable tokenized tensors
├── src/
│   ├── tokenizer/           # BPE training, encoding, and decoding
│   ├── datasets/            # Loading, preprocessing, sampling, and collation
│   ├── model/               # GPT model components
│   ├── optim/               # Optimizers, schedulers, and EMA
│   ├── training/            # Training, evaluation, metrics, and checkpoints
│   ├── post_training/       # Preference data and DPO objective
│   ├── evaluation/          # Held-out capability benchmark scoring
│   ├── inference/           # Generation, sampling, and KV caching
│   ├── mcp/                 # MCP stdio client and tool schemas
│   ├── serving/             # FastAPI and WebSocket interfaces
│   └── utils/               # Configuration, devices, logging, and seeds
├── checkpoints/             # Resumable training state
├── exports/                 # SafeTensors, ONNX, and GGUF artifacts
├── scripts/                 # Command-line workflows
└── tests/                   # Unit and integration tests
```

## Model architecture

The current model is a flexible, modern GPT-style network featuring:

1. Learned or Rotary (RoPE) Positional Embeddings (with Linear Position Interpolation support)
2. Token Embedding matrix with configurable initialization strategies (`normal`, `mean`, `zero`) for vocabulary resizing
3. Multi-Head Self-Attention with Grouped-Query Attention (GQA/MQA) support
4. Gradient Checkpointing for memory-efficient long-sequence training
5. GELU / SwiGLU / GEGLU feed-forward networks
6. Layer Normalization / RMSNorm and pre-norm residual connections
7. Causal Dot-Product Attention with cached causal masking and PyTorch SDPA integration
8. Vocabulary projection head with optional weight tying

Architecture values live in model configurations. The active v2 paths use
[`configs/model.v2.gpu.yaml`](configs/model.v2.gpu.yaml) for the 54.4M-parameter
GPU model and [`configs/model.v2.cpu.yaml`](configs/model.v2.cpu.yaml) for the
15.8M-parameter compact CPU model. The unversioned files remain available for
legacy v1 experiments.

### Token embedding matrix

`TokenEmbedding` maps vocabulary IDs into the model's hidden dimension. It uses
GPT-style normal initialization, keeps the padding row zero, optionally scales
vectors by the square root of the hidden size, and supports freezing, safe
vocabulary resizing, hardware-aligned vocabulary padding, and weight sharing
with the language-model output projection. Device and floating-point dtype can
be selected at construction time.

With the active v2 GPU configuration, the embedding matrix contains
`32,000 × 512 = 16.384M` parameters and is tied to the language-model output
projection. The v2 CPU profile uses `32,000 × 256 = 8.192M`. Initialization and behavior are controlled by
`padding_idx`, `initializer_range`, `scale_embeddings`, and
`freeze_embeddings` in files such as
[`configs/model.v2.gpu.yaml`](configs/model.v2.gpu.yaml).

### Causal self-attention

`MultiHeadAttention` uses a fused projection to create query, key, and value
tensors, splits them across attention heads, applies scaled dot-product
attention, and combines the heads through an output projection. Causal masking
is enabled by default, so a token cannot read future training tokens.

The implementation uses PyTorch's optimized scaled-dot-product attention kernel
when available and retains a numerically stable fallback. It supports boolean
padding masks, additive attention biases, training dropout, and incremental
key/value caches shaped as `[batch, heads, sequence, head_dim]` for generation.

### Feed-forward network

`FeedForward` applies the same nonlinear transformation independently to every
token position. The current configurations expand each 256-dimensional token
to 1,024 dimensions, apply GELU, project back to the model dimension, and
then apply dropout.

The implementation also supports ReLU, SiLU, tanh-approximated GELU, SwiGLU,
and GEGLU. Gated variants use one fused gate/value projection. Hidden dimensions
can be rounded to a configurable multiple for accelerator efficiency, while
biases, initialization, dropout, device, and dtype remain configurable.

### Residual connections and normalization

Transformer blocks use pre-normalization by default:

```text
x = x + Attention(Norm(x))
x = x + FFN(Norm(x))
```

This keeps an unobstructed residual path through deep networks and generally
provides more stable optimization than the previous post-norm layout. Each
branch supports independent residual dropout and configurable residual scaling.

The project provides a bias-configurable LayerNorm matching PyTorch's reference
equation and an RMSNorm alternative with float32 accumulation for FP16/BF16
inputs. Normalization type, epsilon, bias, residual layout, dropout, and scale
are configured in model configuration files (for example,
[`configs/model.v2.gpu.yaml`](configs/model.v2.gpu.yaml) or
[`configs/model.v2.cpu.yaml`](configs/model.v2.cpu.yaml)). A final normalization
is applied before the language-model output head.

### Causal language-model loss

`CausalLanguageModelLoss` computes next-token cross-entropy by aligning each
logit position with the following token label. Padding and other excluded
positions use `ignore_index` or an explicit loss mask. The calculation is
performed in FP32 for stable mixed-precision training and returns a
differentiable zero for fully masked batches instead of producing NaN.

Optional label smoothing and logit z-loss are available for regularization and
large-scale numerical stability. Detailed outputs expose cross-entropy, z-loss,
and the valid token count for correctly weighted logging. Perplexity is computed
as `exp(mean_token_cross_entropy)`.

## Model serving

The FastAPI service exposes separate liveness and readiness checks, validated
generation requests, structured errors, request IDs, and WebSocket token
streaming. Its runtime bounds concurrent generations, limits queue wait time,
cancels requests that exceed their deadline, and runs backend startup/shutdown
hooks through the application lifespan.

```text
GET  /health/live
GET  /health/ready
POST /v1/generate
WS   /v1/generate/stream
GET  /v1/models
POST /v1/chat/completions
```

Generation requests accept `response_format: "plain"` (the default) or
`response_format: "markdown"`. The browser playground exposes the same setting.
The native endpoint and playground also accept up to eight UTF-8 text/code
attachments (256 KiB each). Attachment text is bounded before it enters the
model prompt and is explicitly marked as untrusted reference data.
They also accept `mode: "balanced"`, `"creative"`, `"precise"`, or `"coding"`;
the playground mode picker starts a fresh conversation whenever the mode changes.
The `tools` array supports `"calculator"` and `"datetime"`, and the playground
offers those alongside web search and allowlisted MCP servers in its Tools menu. `/calc`, `/time`, and
`/search` shortcuts activate the corresponding tool directly.
Set `web_search: true` (or prefix the prompt with `/search`) to retrieve SearXNG
or Brave results before generation. The response always includes source URLs.

Local-document RAG is also available. Build an index from text, Markdown,
HTML, CSV/TSV, JSON/JSONL, YAML, or PDF files (PDF support uses the optional
`rag` dependency):

Copy your knowledge files into `documents/` first; the repository includes a
small README there so the command can be tested immediately.

```bash
.venv/bin/pip install -e '.[rag]'
.venv/bin/python scripts/build_rag_index.py docs/ \
  --output data/rag/index.sqlite
```

For broad project coverage plus private documents, use:

```bash
.venv/bin/python scripts/build_rag_index.py README.md docs/ documents/ \
  --output data/rag/index.sqlite
```

Download a bounded, attributable multilingual Wikipedia RAG corpus (10,000
articles each from Simple English, Bengali, and Hindi by default):

```bash
.venv/bin/python scripts/download_rag_dataset.py
```

Add general English coverage without replacing previously downloaded language
files or manifest counts:

```bash
.venv/bin/python scripts/download_rag_dataset.py \
  --languages en --articles-per-language 20000
```

The downloader streams Parquet row groups, deletes temporary shards, checks
free disk space, and records the Wikimedia snapshot and CC BY-SA/GFDL license
metadata. SQLite FTS5 indexing also streams chunks to disk, so the complete
corpus does not need to fit in RAM:

```bash
.venv/bin/python scripts/build_rag_index.py README.md docs/ documents/ \
  data/rag/wikipedia/ --output data/rag/index.sqlite
```

Set `rag.enabled: true` in `configs/inference.v2.yaml`, restart the server, and
send `{"prompt":"What does the guide say?","rag":true}` to `/v1/generate`.
The `/rag What does the guide say?` shortcut works with the native API,
OpenAI-compatible chat UIs, and the browser playground. Retrieved text is
treated as untrusted, answers are instructed to cite `[1]`, and responses show
document/chunk sources. Enable both `web_search` and `rag`, or use `/hybrid`,
to combine current web results with private local knowledge in one grounded
prompt.

Start the server with:

```bash
python scripts/serve.py --host 0.0.0.0 --port 8000
```

Open `http://127.0.0.1:8000/ui/` for the built-in browser playground. It can
check readiness, call the REST endpoint, stream tokens over WebSocket, adjust
sampling parameters, stop an active request, and display token usage. When API
key authentication is enabled, use non-streaming mode because browser
WebSockets cannot attach the required `Authorization` header.

Interactive Swagger documentation is served at `http://127.0.0.1:8000/docs`,
ReDoc at `http://127.0.0.1:8000/redoc`, and the raw OpenAPI schema at
`http://127.0.0.1:8000/openapi.json`.

`/v1/models` and `/v1/chat/completions` provide text-only OpenAI Chat
Completions compatibility for third-party clients such as Open WebUI. Both JSON
and SSE streaming are supported. Configure the client base URL as
`http://HOST:8000/v1`, select model `gopi-v2`, and provide `GOPI_API_KEY` as its
API key. See the v2 usage guide for limits and a complete example.

V2 serving defaults come from
[`configs/inference.v2.yaml`](configs/inference.v2.yaml). Set
`GOPI_CHECKPOINT_PATH=checkpoints/v2-dpo/best.pt` (or the best SFT checkpoint)
when serving the final assistant; the file's conservative default points to the
pretraining checkpoint. Model, tokenizer, and runtime settings can be overridden
with `GOPI_MODEL_CONFIG`, `GOPI_TOKENIZER_PATH`, `GOPI_CHECKPOINT_PATH`,
`GOPI_DEVICE`, `GOPI_MODEL_NAME`, `GOPI_BOT_NAME`,
`GOPI_MAX_CONCURRENCY`, `GOPI_QUEUE_TIMEOUT_SECONDS`,
`GOPI_GENERATION_TIMEOUT_SECONDS`, and `GOPI_CORS_ORIGINS`.
Set `GOPI_API_KEY` to require bearer authentication and
`GOPI_REQUESTS_PER_MINUTE` to enforce the built-in per-process safety limit.
The `/metrics` endpoint exposes request, failure, concurrency, and generation
time counters for scraping or gateway integration.

An optional authenticated workspace endpoint supports bounded coding-agent
operations: repository read/search, previewed hash-checked edits, checked
patches, allowlisted pytest presets, and read-only Git status/diff/log. Enable
it only for a trusted local workspace with `GOPI_WORKSPACE_AGENT_ENABLED=true`
and `GOPI_WORKSPACE_ROOT=/absolute/project/path`. It is disabled by default,
requires `GOPI_API_KEY`, provides no arbitrary shell, and never stages or
commits changes. See [the API reference](docs/API_REFERENCE.md#workspace-agent).

Serving can reuse exact prompt-prefill KV state through a bounded prefix cache
and fixed-page allocator. `continuous_streams` multiplexes active token streams;
`Generator.generate_batch()` performs shared tensor prefills/decodes for
equal-length prompt cohorts and continuously removes completed KV-cache rows.
`GOPI_REPLICA_DEVICES=cuda:0,cuda:1` creates independent replicas and routes
requests to the least-active ready replica. SQLite-backed rate limiting lets
multiple server processes share one request window.

Authenticated WebSockets accept an `Authorization: Bearer ...` header or the
browser-compatible `Sec-WebSocket-Protocol: bearer, <key>` form. With
`GOPI_API_KEY` configured, `POST /v1/admin/reload` warms a replacement backend,
atomically routes new traffic to it, drains old in-flight requests, and then
shuts down the previous backend.

At startup the application loads the configured tokenizer, model configuration,
and checkpoint. It reports `503 not_ready` when an artifact is absent or the
backend cannot be loaded. Authentication, TLS, and global rate limiting should
be enforced by the deployment gateway or ingress layer.

## MCP client

`src/mcp` provides an asynchronous Model Context Protocol client for local
stdio servers. It launches commands without a shell, negotiates current MCP or
the legacy initialization handshake, discovers paginated tools, invokes tools,
captures bounded stderr diagnostics, enforces timeouts, and closes child
processes cleanly.

Define reviewed servers in [`configs/mcp.yaml`](configs/mcp.yaml). Commands in
this file execute with the permissions of the LLM Engine process, so never add
an untrusted package, executable, argument, working directory, or environment
value. List a server's tools and make an explicit call with:

```bash
python scripts/mcp_client.py list --server filesystem
python scripts/mcp_client.py call --server filesystem read_file '{"path":"README.md"}'
```

The configuration includes templates for all seven maintained MCP reference
servers: Filesystem, Memory, Sequential Thinking, Everything, Fetch, Git, and
Time. Only the read-only subset of Filesystem is enabled. The others are
disabled until their dependencies, permissions, and tool schemas are reviewed.
Their templates include conservative allowlists. The TypeScript servers use
`npx`; Fetch, Git, and Time
use `uvx` and require [uv](https://docs.astral.sh/uv/) to be installed.

To inspect a disabled server safely, first install its runner, temporarily set
`enabled: true`, and review its `allowed_tools`; backend startup intentionally
skips servers with empty allowlists. The standalone CLI
can list configured server tools without enabling model access.

The CLI is intentionally explicit and does not let generated model text launch
tools automatically. The serving backend can opt into model-routed calls by
setting `mcp: true` on a generation request. It performs a private JSON planning
pass, validates the selected server and tool against `allowed_tools`, calls the
server, labels its result as untrusted external data, and gives that result to
the model for the visible answer:

```json
{
  "prompt": "Read README.md and summarize the project",
  "mcp": true,
  "mcp_server": "filesystem"
}
```

MCP is disabled per request unless `mcp: true` is supplied. The example
filesystem policy contains read-only tools. Do not add mutating tools such as
`write_file` or `edit_file` without an application-level approval step. Tool
routing quality also depends on the model having learned the required compact
JSON format; invalid or non-allowlisted plans are ignored safely.

For deterministic routing that does not depend on model quality, send an
explicit prompt while keeping `mcp: true`:

```text
/mcp filesystem read_text_file {"path":"README.md"}
```

## Assistant identity

The assistant is named **Gopi**. Its default identity is configured in
[`configs/inference.v2.yaml`](configs/inference.v2.yaml):

```yaml
bot_name: Gopi
system_prompt: You are Gopi, a helpful, honest, and friendly AI assistant.
```

Processed conversational records also contain `"bot_name": "Gopi"` and the
same system message.

## Datasets

### WikiText-103 Raw

WikiText-103 provides general English text for language-model pretraining.

- Source: [Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext)
- Local raw path: `data/raw/wikitext-103-raw-v1/`
- Format: Parquet

Convert the raw line-oriented Parquet shards into article-level JSONL:

```bash
.venv/bin/python scripts/prepare_wikitext.py
```

The generated train and validation splits are used by the pretraining and
combined-training profiles. Supervised fine-tuning profiles use conversation
and instruction datasets instead.

### UltraChat 200k subset

UltraChat supplies multi-turn instruction conversations for supervised
fine-tuning. The local working subset contains:

- 20,000 training conversations
- 2,000 validation conversations
- 2,000 test conversations
- License: MIT
- Source: [HuggingFaceH4/ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k)

Regenerate the processed subset with:

```bash
.venv/bin/python scripts/prepare_ultrachat.py
```

### DailyDialog subset (excluded from active profiles)

DailyDialog adds human-written, everyday conversations. The processed subset
contains 2,000 user/assistant pairs.

- License: CC BY-NC-SA 4.0
- Source: [ConvLab/dailydialog](https://huggingface.co/datasets/ConvLab/dailydialog)
- Commercial-use warning: this dataset is non-commercial and share-alike.

DailyDialog is retained only as an optional experiment and provenance example.
It is not part of the active pretraining or fine-tuning configurations.

Regenerate it with:

```bash
.venv/bin/python scripts/prepare_dailydialog.py
.venv/bin/python scripts/split_dailydialog.py
```

### Hugging Face Dataset Downloader

Download and process compatible Hugging Face datasets into raw Parquet
(`data/raw/<dataset>/`) and chat JSONL (`data/processed/<dataset>/`). Downloading
a dataset does not make its license or contents suitable for training; review
its license, provenance, privacy, safety, and schema first.

```bash
# Download Databricks Dolly 15k
python scripts/prepare_hf_dataset.py --dataset databricks/databricks-dolly-15k --train-size 5000 --validation-size 500

# Download HuggingFace NoRobots
python scripts/prepare_hf_dataset.py --dataset HuggingFaceH4/no_robots --train-size 5000 --validation-size 500
```

Raw data and generated training artifacts are intentionally excluded from Git.
Dataset-specific provenance is documented under `data/processed/<dataset>/README.md`.

The v2 SFT profiles additionally expect category-specific datasets for coding,
general QA, safety, writing/editing, Bengali and Hindi, and tool calling. The
bounded local copies used by the profiles can be reproduced with:

```bash
python scripts/prepare_hf_dataset.py --full --dataset iamtarun/python_code_instructions_18k_alpaca --output-dir data/processed/code_instructions
python scripts/prepare_hf_dataset.py --full --dataset databricks/databricks-dolly-15k --output-dir data/processed/general_qa
python scripts/prepare_hf_dataset.py --full --dataset fwnlp/self-instruct-safety-alignment --output-dir data/processed/safety_alignment
python scripts/prepare_hf_dataset.py --full --dataset HuggingFaceH4/no_robots --output-dir data/processed/writing_editing
python scripts/prepare_hf_dataset.py --full --dataset rishiraj/bengalichat --output-dir data/processed/multilingual_bn_hi
python scripts/prepare_hf_dataset.py --full --dataset rishiraj/hindichat --output-dir data/processed/multilingual_hi
python scripts/prepare_hf_dataset.py --full --dataset narrative-io/narrative-function-calling-v1 --output-dir data/processed/tool_calling
```

These sources have different licenses and provenance characteristics. Review
their current dataset cards and add reviewed manifests before changing the v2
governance policy from `warn` to `error` or enabling commercial use.

The v2 SFT profiles use `dataset_weights` as target dataset-level sampling
shares. A dataset's configured weight is divided across its rows, so a large
source such as OpenOrca does not dominate a smaller capability dataset merely
because it contains more records. `samples_per_epoch` bounds training time and
sampling is deterministic for a given seed and epoch. Validation remains an
unweighted pass over every configured validation record.

Chat records use assistant-only supervision: system and user tokens remain in
the input context but are masked from the causal-language-model loss. Plain
text records continue to train on every non-padding token.

### Dataset governance audit

Machine-readable `dataset-manifest.yaml` files record the source, version,
license review state, commercial-use decision, allowed training stages, and
privacy-review state. Audit every dataset referenced by a training profile:

```bash
python scripts/audit_datasets.py \
  --training-config configs/finetuning.v2.gpu.yaml --stage sft
```

The command exits unsuccessfully for missing, invalid, unreviewed, or
policy-incompatible manifests and emits JSON suitable for CI. Training profiles
also define `dataset_governance.policy`: `warn` reports findings while allowing
an educational run, `error` blocks training, and `off` explicitly disables the
check. Set `commercial_use: true` to require an affirmative commercial-use
decision. A manifest records a review; it does not replace reading the actual
license or conducting a privacy assessment.

## Tokenizer

The tokenizer is a reversible byte-level BPE implementation built in this
repository. It provides full UTF-8 coverage without unknown characters,
Unicode-aware pre-tokenization, deterministic incremental merge training,
bounded encoding caches, explicit control-token handling, and atomic artifact
writes.

Reserved tokens are:

```text
<|pad|> <|unk|> <|bos|> <|eos|> <|system|> <|user|> <|assistant|>
```

Train the configured 50,000-token vocabulary from the local datasets:

```bash
python scripts/tokenize.py train
```

The command creates `tokenizer.json`, `vocab.json`, and `merges.txt` under
`data/tokenizer/`. Inspect an artifact with an exact encode/decode round trip:

```bash
python scripts/tokenize.py inspect "Hello Gopi!" --add-bos --add-eos
```

Training size, minimum pair frequency, input sources, special tokens, and output
location are controlled by [`configs/tokenizer.yaml`](configs/tokenizer.yaml).

## Environment setup

Python 3.10 or newer is required. Python 3.12 is used during current development.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --editable '.[dev]'
```

The editable installation includes PyTorch, PyYAML, PyArrow, SafeTensors,
FastAPI, Uvicorn, and pytest.

## Configuration

Configuration is divided by responsibility:

| File | Responsibility | Key Parameters |
| --- | --- | --- |
| `configs/model.v2.cpu.yaml` | Active compact v2 CPU architecture | 15.8M parameters, 32K vocabulary, hidden size 256, 8 layers, GQA, RoPE, RMSNorm, SwiGLU |
| `configs/model.v2.gpu.yaml` | Active laptop-GPU v2 architecture | 54.4M parameters, 32K vocabulary, hidden size 512, 10 layers, GQA, RoPE, RMSNorm, SwiGLU |
| `configs/model.v2.55m-source.yaml` | Frozen source shape for checkpoint growth | Original v2 GPU shape: 32K base vocabulary, hidden size 512, 10 layers |
| `configs/model.v3.gpu.yaml` | Grown laptop-GPU architecture | 79.3M parameters, 36K vocabulary, hidden size 512, 16 layers |
| `configs/pretraining.v2.cpu.yaml` | Active CPU v2 pretraining | TinyStories/WikiText 35/65, 128 tokens, effective batch 32, 3 epochs, FP32 |
| `configs/pretraining.v2.gpu.yaml` | Active GPU v2 pretraining | TinyStories/WikiText 35/65, 256 tokens, effective batch 32, FP16, 3 epochs |
| `configs/pretraining.v2.continued.gpu.yaml` | Optional WikiText-heavy new training stage | TinyStories/WikiText 15/85, one epoch, separate metric and checkpoints |
| `configs/pretraining.v2.packed.cpu.yaml` | CPU v2 pretraining from memory-mapped token shards | TinyStories/WikiText 35/65 weighted validation, 256-token packed sequences |
| `configs/pretraining.v2.packed.gpu.yaml` | GPU v2 pretraining from memory-mapped token shards | Same objective and weights as JSONL v2, FP16, runtime tokenization removed |
| `configs/pretraining.v3.grown.gpu.yaml` | Continued pretraining after 55M-to-79M growth | Batch 1, effective batch 32, FP16, 500K samples/epoch, fresh optimizer |
| `configs/finetuning.v2.cpu.yaml` | Quality-balanced CPU v2 SFT | Chat, factual, reasoning, Bengali, Hindi, and coding; response-only loss |
| `configs/finetuning.v2.gpu.yaml` | Active quality-balanced GPU v2 SFT | 18 datasets, 500K samples/epoch, 384 tokens, effective batch 32, BF16, 3 epochs |
| `configs/finetuning.v2.packed.cpu.yaml` | CPU v2 SFT from response-masked shards | Same quality mixture with runtime tokenization removed |
| `configs/finetuning.v2.packed.gpu.yaml` | GPU v2 SFT from response-masked shards | 384-token packed sequences, BF16, weighted domain validation |
| `configs/dpo.v2.cpu.yaml` | Single-device CPU preference training | chosen/rejected pairs, batch size 1, 256 tokens, 2 epochs |
| `configs/dpo.v2.gpu.yaml` | Single-GPU FP16 preference training | chosen/rejected pairs, batch size 1, 256 tokens, 2 epochs |
| `configs/tokenizer.v2.yaml` | V2 base-tokenizer training and append-only extension setup | `vocab_size: 32000`, balanced source sampling, extension sources and discovery limits |
| `configs/tokenizer.v3.extension.yaml` | Second verified append-only extension | Preserves the current 34K IDs and discovers up to 2,000 additional tokens |
| `configs/evaluation.v2.pretraining.yaml` | Pretraining domain evaluation | TinyStories and WikiText reported independently |
| `configs/evaluation.v2.finetuning.yaml` | SFT domain evaluation | English, Bengali, Hindi, coding, GSM8K, and chat |
| `configs/inference.v2.yaml` | Active v2 inference and serving defaults | Gopi identity, sampling, context memory, model paths, concurrency, cache, and rate limits |

Legacy v1 profiles remain under the unversioned `model.*`, `pretraining.*`,
`finetuning.*`, `training.*`, `tokenizer.yaml`, and `inference.yaml` names. New
v2 work should use the explicit `.v2.` files and the guides linked in Quick
start.

Training code must not read inference settings, and inference code must not
depend on the training package.

### From-scratch v2 model

The v2 profiles provide the active from-scratch path for memory-constrained
hardware. The 54.4M-parameter GPU architecture uses a 32K vocabulary, 512
hidden dimensions, 10 layers, rotary positions, RMSNorm,
SwiGLU, grouped-query attention, tied embeddings, and gradient checkpointing.
It does not load or depend on third-party pretrained weights.

Prepare the dedicated tokenizer before training v2. This intentionally creates
`data/tokenizer-v2` and does not overwrite the tokenizer used by v1 checkpoints:

```bash
python scripts/tokenize.py train --config configs/tokenizer.v2.yaml
```

The v2 tokenizer uses `source_sampling: balanced_bytes`: it repeatedly reads
from the source that has contributed the fewest UTF-8 bytes. This prevents the
first large corpus from exhausting `max_training_bytes` before later chat,
code, and multilingual sources are represented. Regenerating a tokenizer
changes token IDs and therefore requires model training to restart from
scratch. New checkpoints record a tokenizer fingerprint and reject a different
same-size tokenizer during training and inference.

After pretraining, do not retrain the tokenizer merely to add domain terms.
Create an append-only extension instead; existing IDs and BPE merge priorities
remain unchanged, while Bengali/Hindi text, emoji sequences, phrases, and code
literals can receive new IDs:

```bash
.venv/bin/python scripts/tokenize.py extend \
  --tokenizer data/tokenizer-v2 \
  --output data/tokenizer-v2-extended \
  --token " বাংলা" \
  --token "domainTerm" \
  --token "👨‍👩‍👧‍👦"
```

For a larger list, use `--tokens-file new-tokens.txt` (one UTF-8 token per
line). By default the command also writes to a sibling `-extended` directory,
leaving the base tokenizer intact. It records the base fingerprint and appends
IDs without moving old ones. Start continued pretraining as a new optimizer stage with
`--init-from`, not `--resume`; the loader copies all checkpoint vocabulary rows
and randomly initializes only the appended rows. When the model YAML still has
the base vocabulary size, `scripts/train.py` recognizes verified tokenizer
lineage and selects the extended size automatically.

The v2 configuration also defines bounded automatic discovery across the new
emoji, Bengali, Hindi/Hinglish, and coding datasets. It ranks frequent terms by
the number of existing tokens they replace and appends at most 2,000 IDs:

```bash
.venv/bin/python scripts/tokenize.py extend --config configs/tokenizer.v2.yaml
```

`max_scan_bytes` bounds how much source text is inspected, `min_frequency`
removes rare candidates, `min_existing_tokens` requires a candidate to replace
at least that many base-tokenizer IDs, and `max_new_tokens` caps vocabulary
growth. Dataset files remain training inputs through the fine-tuning profiles;
extension discovery only improves their tokenization efficiency.

Continue from the pretrained weights into new checkpoint files. Do not replace
the original tokenizer or pretraining checkpoints:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/python scripts/train.py \
  --model-config configs/model.v2.gpu.yaml \
  --training-config configs/pretraining.v2.gpu.yaml \
  --tokenizer data/tokenizer-v2-extended \
  --init-from checkpoints/v2-pretraining/best.pt \
  --output checkpoints/v2-tokenizer-extended/latest.pt \
  --best-output checkpoints/v2-tokenizer-extended/best.pt
```

The resulting model vocabulary is `32000 + added_vocab_size`. The model loader
derives that size from verified tokenizer lineage, copies rows `0..31999`
exactly, initializes only appended embedding/output rows, and preserves tied
input/output weights.

### Grow the 55M v2 checkpoint to the 79M v3 model

This optional workflow reuses a trained 10-layer v2 GPU model while adding six
transformer layers and extending its vocabulary from 34K to 36K. It is not a
normal resume: `scripts/grow_checkpoint.py` creates a new model checkpoint with
a fresh training state. It copies layers `0..9` and vocabulary rows `0..33999`
exactly, mean-initializes the new vocabulary rows, and makes layers `10..15`
identity-like by zeroing their attention and feed-forward output projections.
The new blocks therefore disturb the learned residual stream as little as
possible before continued pretraining.

Keep the original tokenizer and checkpoint. The 36K tokenizer must be a second
append-only extension of `data/tokenizer-v2-extended`; a newly trained 36K
tokenizer is not compatible because it can assign different IDs.

First create the verified 34K-to-36K tokenizer extension:

```bash
.venv/bin/python scripts/tokenize.py extend \
  --config configs/tokenizer.v3.extension.yaml
```

Confirm that the command reports an old vocabulary size of 34,000 and a new
size of 36,000. If discovery finds fewer than 2,000 acceptable tokens, do not
run the converter with a 36K target configuration; adjust the extension corpus
or use a target model configuration matching the actual tokenizer size.

Convert the selected v2 fine-tuning checkpoint on CPU:

```bash
.venv/bin/python scripts/grow_checkpoint.py \
  --checkpoint checkpoints/v2-finetuning/best.pt \
  --source-model-config configs/model.v2.55m-source.yaml \
  --target-model-config configs/model.v3.gpu.yaml \
  --source-tokenizer data/tokenizer-v2-extended \
  --target-tokenizer data/tokenizer-v2-extended-36k \
  --output checkpoints/v3-grown/init.pt
```

The converter rejects unrelated tokenizers, shrinking vocabularies, targets
that are not deeper, and incompatible hidden dimensions. It intentionally does
not copy optimizer, scheduler, scaler, sampler, or EMA state. Pass `--use-ema`
only when the EMA weights are deliberately preferred over the live checkpoint
weights.

Continue causal-language-model pretraining with a fresh optimizer:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/python scripts/train.py \
  --model-config configs/model.v3.gpu.yaml \
  --training-config configs/pretraining.v3.grown.gpu.yaml \
  --tokenizer data/tokenizer-v2-extended-36k \
  --init-from checkpoints/v3-grown/init.pt \
  --output checkpoints/v3-pretraining/latest.pt \
  --best-output checkpoints/v3-pretraining/best.pt
```

The 4 GB GPU profile uses batch size 1, gradient accumulation 32, FP16,
gradient checkpointing, a `5e-5` peak learning rate, and a 5% warmup. Watch
validation loss and `nonfinite_updates`; retain the original v2 model until the
grown checkpoint has demonstrated better held-out and generation quality.
After continued pretraining stabilizes, run SFT again with a separate v3
fine-tuning profile before optional DPO.

### Live training report

`scripts/train.py` appends its existing structured log messages to
`logs/training.log` by default and launches `scripts/build_training_report.py`
as an isolated subprocess on rank zero. Training does not wait for report
parsing or JSON generation. The reporter does not import the trainer, load
model tensors, use GPU memory, or modify checkpoints; failure of the reporter
does not stop training. It exits automatically when its parent training
process ends.

The reporter incrementally reads only newly appended complete log lines and
atomically refreshes `reports/training_report.json` every five seconds. This JSON
contains compact training history, aggregate and per-domain validation,
checkpoint file metadata, warnings, configuration snapshots, and derived
progress analysis. The analysis classifies the latest trend as `improving`,
`plateau`, `worsening`, or `waiting_for_validation`, with absolute and
percentage changes for overall loss, recent validation, perplexity, training
loss, and each validation domain.

Run training normally; no `tee` command or separate report-builder command is
required:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/python scripts/train.py \
  --model-config configs/model.v2.55m-source.yaml \
  --training-config configs/finetuning.v2.gpu.yaml \
  --tokenizer data/tokenizer-v2-extended \
  --resume checkpoints/v2-finetuning/latest.pt \
  --output checkpoints/v2-finetuning/latest.pt \
  --best-output checkpoints/v2-finetuning/best.pt
```

Serve the static viewer from a second terminal:

```bash
.venv/bin/python -m http.server 8000 --directory reports
```

Open `http://localhost:8000/training_report.html`. The page loads
`training_report.json` by default and refreshes every five seconds. It displays
training and validation loss, per-domain loss, perplexity, learning rate,
gradient norm, throughput, memory/checkpoint details, improvement tables,
warnings, and the recent raw log tail. It also highlights possible overfitting,
ranks domains by current validation loss, compares latest and best validation
checkpoints, summarizes the run, and identifies additional evaluations that
are not available from trainer logs (dataset quality, fixed-prompt generation
quality). The independent reporter also samples live CPU utilization, system
and training-process RAM, GPU utilization, VRAM, temperature, power draw,
power limit, and fan speed. NVIDIA fields gracefully show as unavailable when
`nvidia-smi` or a particular sensor is unsupported. Up to 3,600 telemetry
samples are retained by default. Hardware is sampled every five seconds while
the log is still checked every report-refresh interval. The reporter writes
compact JSON atomically and skips writes when neither logs nor telemetry have
changed, reducing disk and external-drive activity. Use `--telemetry-points`
and `--telemetry-seconds` to tune retention and sampling frequency.
Query
parameters can select another
JSON file or browser refresh interval:

```text
http://localhost:8000/training_report.html?data=experiment-2.json&refresh=1000
```

Useful training options are:

```text
--log-file logs/experiment-2.log
--report-json reports/experiment-2.json
--report-refresh-seconds 5
--report-telemetry-seconds 5
--report-telemetry-points 3600
--no-live-report
```

The last option disables automatic reporter startup. The builder can also run
independently; it watches every five seconds by default, while
`--watch-seconds 0` performs a single conversion:

```bash
.venv/bin/python scripts/build_training_report.py \
  --log logs/experiment-2.log \
  --output reports/experiment-2.json
```

Only `reports/training_report.html` is version-controlled. Generated report
JSON, sample data, raw logs, and graph artifacts are ignored by Git and remain
local to the experiment.

Run staged GPU training with separate checkpoints:

```bash
python scripts/train.py \
  --model-config configs/model.v2.gpu.yaml \
  --training-config configs/pretraining.v2.gpu.yaml \
  --tokenizer data/tokenizer-v2 \
  --output checkpoints/v2-pretraining/latest.pt \
  --best-output checkpoints/v2-pretraining/best.pt

python scripts/train.py \
  --model-config configs/model.v2.gpu.yaml \
  --training-config configs/finetuning.v2.gpu.yaml \
  --tokenizer data/tokenizer-v2 \
  --init-from checkpoints/v2-pretraining/best.pt \
  --output checkpoints/v2-finetuning/latest.pt \
  --best-output checkpoints/v2-finetuning/best.pt
```

The v2 pretraining corpus uses complete bounded WikiText chunks instead of
silently truncating each article. Base pretraining runs three epochs with a
35/65 TinyStories/WikiText objective. The optional
`configs/pretraining.v2.continued.gpu.yaml` stage runs once with a 15/85 mix and
must write separate checkpoints. SFT samples 500,000 examples per epoch across
18 chat, factual, reasoning, coding, Bengali, Hindi/Hinglish, safety, tool, and
emoji datasets. These additions improve data coverage but do not make a small
from-scratch model equivalent to a billion-parameter model.

### Large-model architecture profiles

Optional architecture targets are provided as `configs/model.future.1b.yaml`,
`configs/model.future.7b.yaml`, and `configs/model.future.30b.yaml`. They verify
that model dimensions, GQA, SwiGLU, RoPE, and long context are configuration
driven; they are not laptop training profiles and do not include the distributed
cluster resources required to train models of those sizes. The engine provides
opt-in FSDP, but these files intentionally contain architecture values only.

Inspect a profile without allocating its weights or risking an out-of-memory
error:

```bash
python scripts/inspect_model.py configs/model.v2.gpu.yaml
python scripts/plan_training.py --model-config configs/model.v2.gpu.yaml --training-config configs/pretraining.v2.gpu.yaml --training-tokens 1000000
python scripts/inspect_model.py configs/model.future.7b.yaml
```

The estimator reports parameter-weight and full-length KV-cache memory. Real
training needs substantially more memory for gradients, optimizer states,
activations, communication buffers, and temporary kernels. The loader also
accepts common configuration aliases such as `num_hidden_layers`,
`num_attention_heads`, `num_key_value_heads`, `context_length`, and
`intermediate_size`, while retaining all existing key names.

### Scale-out training and data

Multi-process DDP remains the default. Large-scale training profiles may opt into
parameter, gradient, and optimizer sharding with:

```yaml
distributed_strategy: fsdp       # or fsdp_hybrid across suitable nodes
mixed_precision: bf16
gradient_checkpointing: true     # model configuration
fused_optimizer: true
```

Launch distributed runs with `torchrun`. FSDP requires CUDA and more than one
process; `fsdp_hybrid` is intended for multi-node jobs. The engine wraps each
Transformer block independently, enables original-parameter optimizer handling,
limits all-gathers, and selects BF16/FP16 FSDP policies from the training file.
Current single-file checkpoints remain the default. Selecting `fsdp` or
`fsdp_hybrid` automatically makes `--output` and `--best-output` reshardable
checkpoint directories and disables parameter EMA to preserve sharded memory.
Other strategies can opt in with `checkpoint_format: distributed`. Cluster launchers can use
`training.distributed_checkpoint.save_distributed_checkpoint` and
`load_distributed_checkpoint` for collective, reshardable model/optimizer
directories on a shared filesystem. Rank-local files preserve scheduler,
mixed-precision scaler, and RNG state. RNG is restored only when the world size
matches; resharded runs restore scheduler/scaler state and begin with the
launcher's seeded RNG streams. Validate save, restart, and world-size changes on
the exact cluster topology before a long run.

An opt-in 1B example is provided in
`configs/pretraining.future.fsdp.yaml`. After building its binary shard input,
launch it on a suitable multi-GPU machine with:

```bash
torchrun --standalone --nproc-per-node=8 scripts/train.py \
  --model-config configs/model.future.1b.yaml \
  --training-config configs/pretraining.future.fsdp.yaml \
  --tokenizer data/tokenizer-v2 \
  --output checkpoints/future-1b/latest \
  --best-output checkpoints/future-1b/best
```

Do not run this profile on the RTX 3050 laptop.

Check accelerator support before selecting precision or fused-kernel options:

```bash
python scripts/capabilities.py
```

PyTorch SDPA already dispatches to an efficient/FlashAttention kernel when the
installed CUDA build, GPU, dtype, and tensor shapes support it. FP8 is reported
separately because having a float8 dtype in PyTorch does not mean the GPU can
execute FP8 training. RTX 3050 hardware should use FP16, not FP8.

For corpora too large for JSONL tokenization during training, build bounded
binary shards with filtering, exact and near deduplication, benchmark
contamination exclusion, English screening, PII redaction, document packing,
and a reproducible manifest:

```bash
python scripts/build_token_shards.py data/processed/wikitext_103/train.jsonl \
  --tokenizer data/tokenizer-v2 \
  --output data/shards/pretraining-v2 \
  --sequence-length 512 --english-only \
  --exclude configs/evaluation.core.jsonl
```

Near duplicates use a bounded 64-bit SimHash index with deterministic FIFO
eviction. `--near-duplicate-distance` controls corpus deduplication (default 3),
while the higher-recall `--contamination-distance` controls benchmark matching
(default 8). Use `-1` to disable either fuzzy check. The shard manifest records
the thresholds, excluded files, capacity, and separate exact-duplicate,
near-duplicate, and contamination rejection counts. Review samples when tuning
thresholds because fuzzy matching can produce false positives.

PII filtering covers email addresses, validated IPv4/IPv6 addresses, phone
numbers, conservative street-address patterns, US Social Security numbers,
Luhn-valid payment-card numbers, checksum-valid IBANs, and common API credential
formats. The manifest reports per-category and total redaction counts. These
deterministic patterns do not identify contextual PII such as arbitrary person
names and should be supplemented with a reviewed NER/DLP system for sensitive
or multilingual corpora.

Set a training profile's `train_files` entries to one or more shard manifests.
The loader memory-maps them on demand, preserves per-dataset mixture weights,
and continues to use rank-aware sampling. Keep benchmark/test data in separate
files; do not include it when building training shards.

For the active 32K v2 laptop model, `auto` storage selects `uint16`, halving
token-shard I/O compared with `uint32`. Build train and validation shards per
dataset so the configured 35/65 TinyStories/WikiText sampling mixture remains
intact:

```bash
.venv/bin/python scripts/build_token_shards.py data/processed/tinystories/train.jsonl \
  --tokenizer data/tokenizer-v2 --sequence-length 256 --workers 6 \
  --output data/shards/pretraining-v2/train/tinystories
.venv/bin/python scripts/build_token_shards.py data/processed/wikitext_103/train.jsonl \
  --tokenizer data/tokenizer-v2 --sequence-length 256 --workers 6 \
  --output data/shards/pretraining-v2/train/wikitext_103
.venv/bin/python scripts/build_token_shards.py data/processed/tinystories/validation.jsonl \
  --tokenizer data/tokenizer-v2 --sequence-length 256 \
  --output data/shards/pretraining-v2/validation/tinystories
.venv/bin/python scripts/build_token_shards.py data/processed/wikitext_103/validation.jsonl \
  --tokenizer data/tokenizer-v2 --sequence-length 256 \
  --output data/shards/pretraining-v2/validation/wikitext_103
```

After all four manifests exist and pass loader validation, train with the packed
profile:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/python scripts/train.py \
  --model-config configs/model.v2.gpu.yaml \
  --training-config configs/pretraining.v2.packed.gpu.yaml \
  --tokenizer data/tokenizer-v2 \
  --init-from checkpoints/v2-pretraining/best.pt \
  --epochs 1 \
  --output checkpoints/v2-packed-continued/latest.pt \
  --best-output checkpoints/v2-packed-continued/best.pt
```

Use `--init-from`, not `--resume`, when switching an existing JSONL run to
packed shards: packing changes sequence and sampler boundaries, so it is a new
continued-pretraining stage with fresh optimizer/scheduler state. Exact resume
within the packed stage can subsequently use
`--resume checkpoints/v2-packed-continued/latest.pt`.

The packed CPU profile reads the same 256-token manifests:

```bash
.venv/bin/python scripts/train.py \
  --model-config configs/model.v2.cpu.yaml \
  --training-config configs/pretraining.v2.packed.cpu.yaml \
  --tokenizer data/tokenizer-v2 \
  --init-from checkpoints/v2-pretraining-cpu/best.pt \
  --output checkpoints/v2-packed-continued-cpu/latest.pt \
  --best-output checkpoints/v2-packed-continued-cpu/best.pt
```

For SFT, ordinary causal shards are incorrect because they would optimize
system and user prompt tokens. Build train and validation data with
`--objective response_only`; every shard then has a parallel binary loss mask
that supervises assistant tokens only. Use the exact tokenizer selected for
fine-tuning because manifests enforce its fingerprint:

```bash
for split in train validation; do
  for dataset in \
    ultrachat_200k helpsteer openorca gsm8k core_chat \
    code_instructions general_qa safety_alignment writing_editing \
    multilingual_bn_hi multilingual_hi tool_calling emoji_chat \
    bangla_qa bangla_reading_qa hindi_history_qa hinglish_chat code_alpaca; do
    .venv/bin/python scripts/build_token_shards.py \
      "data/processed/$dataset/$split.jsonl" \
      --tokenizer data/tokenizer-v2-extended \
      --output "data/shards/finetuning-v2/$split/$dataset" \
      --sequence-length 384 --sequences-per-shard 8192 \
      --objective response_only --workers 4
  done
done
```

After every manifest exists, start packed SFT as a new optimizer stage:

```bash
.venv/bin/python scripts/train.py \
  --model-config configs/model.v2.gpu.yaml \
  --training-config configs/finetuning.v2.packed.gpu.yaml \
  --tokenizer data/tokenizer-v2-extended \
  --init-from checkpoints/v2-pretraining/best.pt \
  --output checkpoints/v2-finetuning-packed/latest.pt \
  --best-output checkpoints/v2-finetuning-packed/best.pt
```

Do not resume a JSONL SFT run into packed data because packing changes sequence
and sampler boundaries. Use `configs/finetuning.v2.packed.cpu.yaml` with
`configs/model.v2.cpu.yaml` for the CPU equivalent. Both packed SFT profiles
consume the same 384-token manifests.

### Preference training and capability evaluation

`src/post_training` provides validated preference data, response-only scoring,
frozen-reference DPO training, validation reward metrics, scheduling, clipping,
mixed precision, early stopping, and checkpoint resume. Input JSONL records use
`prompt`, `chosen`, and `rejected` fields.

Build those preference pairs deterministically from the scored HelpSteer
responses already used for SFT:

```bash
python scripts/prepare_helpsteer_preferences.py
```

```bash
.venv/bin/python scripts/train_dpo.py \
  --model-config configs/model.v2.gpu.yaml \
  --training-config configs/dpo.v2.gpu.yaml \
  --tokenizer data/tokenizer-v2 \
  --reference-checkpoint checkpoints/v2-finetuning/best.pt \
  --init-from checkpoints/v2-finetuning/best.pt \
  --output checkpoints/v2-dpo/latest.pt \
  --best-output checkpoints/v2-dpo/best.pt \
  --device cuda
```

`--init-from` can initialize the policy from another compatible checkpoint;
`--resume` restores its complete training state. The current DPO CLI is
single-device and does not claim distributed support.

Run held-out deterministic generation checks after SFT or preference training:

```bash
python scripts/evaluate_benchmarks.py \
  --checkpoint checkpoints/v2-finetuning/best.pt \
  --model-config configs/model.v2.gpu.yaml \
  --tokenizer data/tokenizer-v2
```

The bundled benchmark reports separate scores for identity, conversation,
instruction following, math, reasoning, knowledge, and safety using the
held-out cases in `configs/evaluation.core.jsonl`.

For v2, report held-out loss and perplexity separately for English, Bengali,
Hindi, coding, GSM8K, and chat data:

```bash
python scripts/evaluate_domains.py \
  --domains configs/evaluation.v2.finetuning.yaml \
  --checkpoint checkpoints/v2-finetuning/best.pt \
  --device cuda
```

The result retains each domain and also includes a token-weighted aggregate.
Test generated answers separately, including Unicode-aware Bengali and Hindi
matching, with the deterministic six-domain smoke suite:

```bash
python scripts/evaluate_benchmarks.py \
  --cases configs/evaluation.v2.domains.jsonl \
  --checkpoint checkpoints/v2-finetuning/best.pt \
  --device cuda
```

These small generation cases are regression smoke tests, not replacements for
larger human-reviewed or established task benchmarks.

During pretraining, evaluate TinyStories and WikiText independently instead of
trusting a token-count aggregate dominated by the larger validation split:

```bash
.venv/bin/python scripts/evaluate_domains.py \
  --domains configs/evaluation.v2.pretraining.yaml \
  --model-config configs/model.v2.gpu.yaml \
  --training-config configs/pretraining.v2.gpu.yaml \
  --tokenizer data/tokenizer-v2 \
  --checkpoint checkpoints/v2-pretraining/best.pt \
  --device cuda
```

V2 training profiles combine fixed domain results using their configured
`validation_weights`; this dataset-weighted score selects the best checkpoint
while every domain remains visible in logs. `validation_metric_name` versions
the score definition, so resuming an older token-weighted checkpoint resets an
incompatible best-loss baseline exactly once.

### Scalable serving

The active generator integrates bounded prefix reuse and optional fixed-page KV
storage. Serving provides continuously admitted stream multiplexing,
least-active replica routing, shared rate limiting, and warm/drain reloads. The
older whole-request `DynamicBatcher` remains available for batch-capable
backends. `Generator.generate_batch()` supplies real tensor-level prefill and
decode batching with KV-row compaction as requests finish. The HTTP stream
scheduler multiplexes independently admitted streams, while tensor batching is
available through the generator API.

`tensor_parallel_size` defaults to 1. Under an initialized `torchrun` process
group, larger values shard attention heads, FFN intermediate channels, and the
vocabulary projection. Attention/FFN outputs use all-reduce and vocabulary
logits use all-gather, so every rank retains the ordinary MiniGPT interface.
Run `scripts/validate_distributed.py` before a job; add
`--tensor-parallel-smoke` to compare sharded and unsharded logits.

For example, a local two-rank validation is:

```bash
torchrun --standalone --nproc-per-node=2 scripts/validate_distributed.py \
  --backend gloo --expected-nodes 1 --tensor-parallel-smoke
```

## Commands

The primary v2 command-line entry points are:

```bash
.venv/bin/python scripts/capabilities.py
.venv/bin/python scripts/tokenize.py train --config configs/tokenizer.v2.yaml
.venv/bin/python scripts/train.py --model-config configs/model.v2.gpu.yaml --training-config configs/pretraining.v2.gpu.yaml --tokenizer data/tokenizer-v2
.venv/bin/python scripts/train_dpo.py --training-config configs/dpo.v2.gpu.yaml --reference-checkpoint checkpoints/v2-finetuning/best.pt --init-from checkpoints/v2-finetuning/best.pt
.venv/bin/python scripts/evaluate_domains.py --domains configs/evaluation.v2.finetuning.yaml --checkpoint checkpoints/v2-finetuning/best.pt --device cuda
.venv/bin/python scripts/generate.py "Hello Gopi" --checkpoint checkpoints/v2-dpo/best.pt --device cuda
.venv/bin/python scripts/export.py --checkpoint checkpoints/v2-dpo/best.pt --format safetensors
.venv/bin/python scripts/serve.py --host 127.0.0.1 --port 8000
```

Training settings, dataset paths, checkpoint frequency, optimizer, scheduler,
and EMA behavior are controlled by the selected YAML profile. Resume only the
same stage from its `latest.pt`; initialize a different stage from the previous
stage's `best.pt`. Exact commands are maintained in the v2 training and usage
guides.
`mixed_precision` accepts `none`, `bf16`, or CUDA-only `fp16`, while
`gradient_accumulation_steps` controls effective batch size.
Training logs and epoch history include learning rate, gradient norm, processed
tokens, token throughput, total progress, elapsed time, ETA, best validation
loss, peak CUDA memory, current allocated/reserved/total GPU memory, and skipped
non-finite updates. New validation minima emit a separate
`new_best_validation` event.
These observability counters are preserved by resumable single-file checkpoints.

### First small training run

Before starting a complete corpus run, verify the pipeline with the compact CPU
model and a single pretraining epoch. The configured inputs are TinyStories and
WikiText, so the required processed files must exist first:

```bash
.venv/bin/python scripts/train.py \
  --model-config configs/model.v2.cpu.yaml \
  --training-config configs/pretraining.v2.cpu.yaml \
  --tokenizer data/tokenizer-v2 \
  --epochs 1 \
  --output checkpoints/v2-cpu-smoke/latest.pt \
  --best-output checkpoints/v2-cpu-smoke/best.pt
```

This profile uses a micro-batch of one, 128-token sequences, 32-step gradient
accumulation, and FP32.

Evaluate a bounded sample without loading the full dataset into memory:

```bash
.venv/bin/python scripts/evaluate.py \
  --model-config configs/model.v2.cpu.yaml \
  --training-config configs/pretraining.v2.cpu.yaml \
  --tokenizer data/tokenizer-v2 \
  --checkpoint checkpoints/v2-cpu-smoke/best.pt \
  --dataset data/processed/tinystories/validation.jsonl \
  --max-batches 25 --device cpu
```

Then verify checkpoint loading and autoregressive generation:

```bash
.venv/bin/python scripts/generate.py "Hello, my name is" \
  --model-config configs/model.v2.cpu.yaml \
  --tokenizer data/tokenizer-v2 \
  --checkpoint checkpoints/v2-cpu-smoke/best.pt \
  --device cpu --raw --max-tokens 40 --temperature 0.8 --seed 42
```

A model trained from scratch on this tiny dataset for one epoch will generally
produce incoherent text; this run validates the pipeline rather than chatbot
quality. Evaluation, generation, interactive chat, serving, benchmarks, and
export load EMA weights by default when a checkpoint contains them. A checkpoint
without EMA transparently uses its ordinary model weights. With
`ema_decay: 0.999`, EMA metrics can lag behind the ordinary model weights during
such a short run.

The small run uses held-out validation and writes recovery and best checkpoints
to the explicit `v2-cpu-smoke` paths above.

### Two-stage CPU training

Start a fresh general-language checkpoint on TinyStories and WikiText. This is
a large CPU job and can take a long time:

```bash
.venv/bin/python scripts/train.py \
  --model-config configs/model.v2.cpu.yaml \
  --training-config configs/pretraining.v2.cpu.yaml \
  --tokenizer data/tokenizer-v2 \
  --output checkpoints/v2-pretraining-cpu/latest.pt \
  --best-output checkpoints/v2-pretraining-cpu/best.pt
```

Initialize a new optimizer and fine-tune the best pretrained weights with the
quality-balanced v2 SFT mixture:

```bash
.venv/bin/python scripts/train.py \
  --model-config configs/model.v2.cpu.yaml \
  --training-config configs/finetuning.v2.cpu.yaml \
  --tokenizer data/tokenizer-v2 \
  --init-from checkpoints/v2-pretraining-cpu/best.pt \
  --output checkpoints/v2-finetuning-cpu/latest.pt \
  --best-output checkpoints/v2-finetuning-cpu/best.pt
```

Use `--resume` only to continue the same training stage with its optimizer,
scheduler, sampler, random-number, and early-stopping state. Use `--init-from`
to transfer model weights into a new stage with a fresh optimizer and schedule.

For a CUDA laptop GPU, use the active v2 profiles. Pretraining uses FP16 and
SFT uses BF16 on a verified-capable GPU:

```bash
.venv/bin/python scripts/train.py \
  --model-config configs/model.v2.gpu.yaml \
  --training-config configs/pretraining.v2.gpu.yaml \
  --tokenizer data/tokenizer-v2 \
  --output checkpoints/v2-pretraining/latest.pt \
  --best-output checkpoints/v2-pretraining/best.pt

.venv/bin/python scripts/train.py \
  --model-config configs/model.v2.gpu.yaml \
  --training-config configs/finetuning.v2.gpu.yaml \
  --tokenizer data/tokenizer-v2 \
  --init-from checkpoints/v2-pretraining/best.pt \
  --output checkpoints/v2-finetuning/latest.pt \
  --best-output checkpoints/v2-finetuning/best.pt
```

The GPU pretraining profile uses micro-batch two, 256-token sequences, and
16-step accumulation. The GPU SFT profile uses micro-batch one, 384-token
sequences, and 32-step accumulation. Do not change precision or tokenizer while
resuming an existing stage.

Run the test suite with:

```bash
.venv/bin/python -m pytest
```

## Current implementation status

| Area | Status |
| --- | --- |
| Local tokenizer, model, training, evaluation and generation | Implemented and tested |
| Token/step/FLOP/memory/runtime/cost training planner | Implemented and tested |
| RoPE, GQA/MQA, RMSNorm, SwiGLU and checkpointed activations | Implemented and tested |
| DDP and rank-aware sampling | Implemented; requires multi-GPU validation |
| FSDP/hybrid FSDP and reshardable checkpoints | Implemented; requires cluster validation |
| Filtered, deduplicated, memory-mapped binary token shards | Implemented and tested locally |
| Dataset provenance/license/privacy policy audit | Implemented; active profiles warn until all reviews are complete |
| SFT and single-device DPO training/checkpoint workflow | Implemented and tested locally |
| Capability benchmark runner | Implemented and tested with the bundled held-out cases |
| REST/WebSocket model serving | Implemented and tested locally |
| Prefix/paged KV reuse, stream multiplexing, replicas, reload, shared limits | Implemented and tested locally |
| Tensor-parallel inference | Implemented; two-rank Gloo equivalence validated locally, NCCL hardware validation remains |
| Multi-node topology/collective validator | Implemented; real cluster validation remains |
| CPU dynamic quantization | Implemented; target-model quality validation remains |
| SafeTensors and PyTorch Export | Implemented |
| ONNX export | Implemented; install `.[export]` and validate in target runtime |
| Automated tests | 311 passing in the current development environment |

## Design principles

- One responsibility per module
- Configuration instead of hardcoded experiments
- Immutable raw data and reproducible preprocessing
- Independent tokenizer and model artifacts
- No dependency from inference to training
- Versioned, resumable checkpoints
- Tests before performance optimization
- Explicit capability and validation boundaries

## License

The project does not currently declare a repository-level software license.
Dataset licenses apply independently; review them before distributing data or
using trained artifacts commercially.
