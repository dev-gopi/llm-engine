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

Architecture values live in model configurations like [`configs/model.gpu.yaml`](configs/model.gpu.yaml) (full model) or [`configs/model.cpu.yaml`](configs/model.cpu.yaml) (compact model), keeping
the model implementation independent from experiment size.

### Token embedding matrix

`TokenEmbedding` maps vocabulary IDs into the model's hidden dimension. It uses
GPT-style normal initialization, keeps the padding row zero, optionally scales
vectors by the square root of the hidden size, and supports freezing, safe
vocabulary resizing, hardware-aligned vocabulary padding, and weight sharing
with the language-model output projection. Device and floating-point dtype can
be selected at construction time.

With the GPU configuration (`configs/model.gpu.yaml`), the matrix contains `50,000 × 256 = 12.8M`
parameters (the CPU configuration currently uses the same embedding dimensions). Its initialization and behavior are controlled by
`padding_idx`, `initializer_range`, `scale_embeddings`, and
`freeze_embeddings` in model configuration files like [`configs/model.gpu.yaml`](configs/model.gpu.yaml).

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
are configured in model configuration files (e.g. [`configs/model.gpu.yaml`](configs/model.gpu.yaml) or [`configs/model.cpu.yaml`](configs/model.cpu.yaml)). A final LayerNorm
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
```

Generation requests accept `response_format: "plain"` (the default) or
`response_format: "markdown"`. The browser playground exposes the same setting.
They also accept `mode: "balanced"`, `"creative"`, `"precise"`, or `"coding"`;
the playground mode picker starts a fresh conversation whenever the mode changes.
The `tools` array supports `"calculator"` and `"datetime"`, and the playground
offers those alongside web search and allowlisted MCP servers in its Tools menu. `/calc`, `/time`, and
`/search` shortcuts activate the corresponding tool directly.
Set `web_search: true` (or prefix the prompt with `/search`) to retrieve SearXNG
or Brave results before generation. The response always includes source URLs.

Start the server with:

```bash
python scripts/serve.py --host 0.0.0.0 --port 8000
```

Open `http://127.0.0.1:8000/ui/` for the built-in browser playground. It can
check readiness, call the REST endpoint, stream tokens over WebSocket, adjust
sampling parameters, stop an active request, and display token usage. When API
key authentication is enabled, use non-streaming mode because browser
WebSockets cannot attach the required `Authorization` header.

Serving defaults come from [`configs/inference.yaml`](configs/inference.yaml)
and can be overridden with `GOPI_MODEL_NAME`, `GOPI_BOT_NAME`,
`GOPI_MAX_CONCURRENCY`, `GOPI_QUEUE_TIMEOUT_SECONDS`,
`GOPI_GENERATION_TIMEOUT_SECONDS`, and `GOPI_CORS_ORIGINS`.
Set `GOPI_API_KEY` to require bearer authentication and
`GOPI_REQUESTS_PER_MINUTE` to enforce the built-in per-process safety limit.
The `/metrics` endpoint exposes request, failure, concurrency, and generation
time counters for scraping or gateway integration.

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
[`configs/inference.yaml`](configs/inference.yaml):

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
| `configs/model.cpu.yaml` | Compact model for CPU development and fresh training | `hidden_size: 256`, `layers: 6`, `heads: 8`, `max_position: 512`, `ffn_hidden_size: 1024` |
| `configs/model.gpu.yaml` | Architecture used by the included trained checkpoint | `hidden_size: 256`, `layers: 8`, `heads: 8`, `max_position: 512`, `ffn_hidden_size: 1024` |
| `configs/pretraining.cpu.yaml` | CPU pretraining profile | TinyStories + WikiText, `batch_size: 2`, `max_sequence_length: 256`, effective batch 8, 10 epochs |
| `configs/pretraining.gpu.yaml` | GPU pretraining profile | TinyStories + WikiText, `batch_size: 2`, `max_sequence_length: 512`, effective batch 32, `mixed_precision: fp16`, 10 epochs |
| `configs/finetuning.cpu.yaml` | CPU supervised fine-tuning profile | UltraChat + HelpSteer + OpenOrca, `batch_size: 2`, `max_sequence_length: 256`, effective batch 32, 3 epochs |
| `configs/finetuning.gpu.yaml` | Memory-safe 4 GB GPU supervised fine-tuning profile | UltraChat + HelpSteer + OpenOrca, `batch_size: 1`, `max_sequence_length: 256`, effective batch 32, gradient checkpointing, `mixed_precision: fp16`, 3 epochs |
| `configs/dpo.v2.cpu.yaml` | Single-device CPU preference training | chosen/rejected pairs, batch size 1, 256 tokens, 2 epochs |
| `configs/dpo.v2.gpu.yaml` | Single-GPU FP16 preference training | chosen/rejected pairs, batch size 1, 256 tokens, 2 epochs |
| `configs/training.cpu.yaml` | Combined CPU profile across retained datasets | TinyStories + WikiText + UltraChat + HelpSteer + OpenOrca, `batch_size: 2`, effective batch 32, 5 epochs |
| `configs/training.gpu.yaml` | Combined GPU profile across retained datasets | TinyStories + WikiText + UltraChat + HelpSteer + OpenOrca, `batch_size: 2`, effective batch 32, `mixed_precision: fp16`, 5 epochs |
| `configs/tokenizer.yaml` | Byte-level BPE tokenizer training setup | `vocab_size: 50000`, `min_frequency: 2`, `special_tokens`, sources list |
| `configs/inference.yaml` | Assistant identity, sampling, and API serving defaults | `bot_name: Gopi`, temperature, top_k/p, context_memory, sqlite session path, concurrency & rate limits |

Training code must not read inference settings, and inference code must not
depend on the training package.

### From-scratch v2 model

The v2 profiles preserve the original trained model while providing a stronger
from-scratch path for memory-constrained hardware. The GPU architecture uses a
32K vocabulary, 384 hidden dimensions, 10 layers, rotary positions, RMSNorm,
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
silently truncating each article. Fine-tuning adds MIT-licensed GSM8K reasoning
examples and a deterministic project-owned core behavior set alongside
UltraChat, HelpSteer, and OpenOrca. These additions improve data coverage but do
not make a small from-scratch model equivalent to a billion-parameter model.

### Large-model architecture profiles

The current v1 and v2 configurations remain unchanged. Optional architecture
targets are provided as `configs/model.future.1b.yaml`,
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
uint32 binary shards with filtering, exact and near deduplication, benchmark
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

Set a training profile's only `train_files` entry to
`data/shards/pretraining-v2/manifest.json`. The loader memory-maps shard files
on demand and continues to use rank-aware sampling. Keep benchmark/test data in
separate files; do not include it when building training shards.

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
python scripts/train_dpo.py \
  --training-config configs/dpo.v2.gpu.yaml \
  --reference-checkpoint checkpoints/v2-finetuning/best.pt
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

The command-line workflows are:

```bash
python scripts/tokenize.py train
python scripts/train.py
python scripts/train_dpo.py --reference-checkpoint checkpoints/v2-finetuning/best.pt
python scripts/evaluate.py
python scripts/generate.py "Hello Gopi"
python scripts/export.py --format safetensors
python scripts/serve.py
python scripts/capabilities.py
python scripts/inspect_model.py configs/model.v2.gpu.yaml
python scripts/evaluate_benchmarks.py --checkpoint checkpoints/v2-finetuning/best.pt
python scripts/audit_datasets.py --training-config configs/pretraining.v2.gpu.yaml --stage pretraining
```

Training settings, dataset paths, checkpoint frequency, optimizer, scheduler,
and EMA behavior are controlled by configuration files such as `configs/training.cpu.yaml` or `configs/training.gpu.yaml`. Training resumes
with `python scripts/train.py --resume checkpoints/latest/model.pt`.
`mixed_precision` accepts `none`, `bf16`, or CUDA-only `fp16`, while
`gradient_accumulation_steps` controls effective batch size.
Training logs and epoch history include learning rate, gradient norm, processed
tokens, token throughput, peak CUDA memory, and skipped non-finite updates.
These observability counters are preserved by resumable single-file checkpoints.

### First small training run

Before starting a complete corpus run, verify the pipeline with the compact CPU
model and a single pretraining epoch. The configured inputs are TinyStories and
WikiText, so the required processed files must exist first:

```bash
python scripts/train.py --model-config configs/model.cpu.yaml \
  --training-config configs/pretraining.cpu.yaml --epochs 1
```

This profile uses two examples per batch, 256-token sequences and four-step
gradient accumulation. It writes the final checkpoint to
`checkpoints/latest/model.pt`.

Evaluate a bounded sample without loading the full dataset into memory:

```bash
python scripts/evaluate.py --model-config configs/model.cpu.yaml \
  --training-config configs/pretraining.cpu.yaml \
  --dataset data/processed/tinystories/validation.jsonl \
  --max-batches 25 --device cpu
```

Then verify checkpoint loading and autoregressive generation:

```bash
python scripts/generate.py "Hello, my name is" \
  --model-config configs/model.cpu.yaml --device cpu \
  --max-tokens 40 --temperature 0.8 --seed 42
```

A model trained from scratch on this tiny dataset for one epoch will generally
produce incoherent text; this run validates the pipeline rather than chatbot
quality. Evaluation loads EMA weights by default. With `ema_decay: 0.999`, EMA
metrics can lag behind the ordinary model weights during such a short run.

The small profile uses a held-out validation split, saves the best validation
checkpoint to `checkpoints/best/model.pt`, and stops after five epochs without
a meaningful validation improvement. To make the best checkpoint available for default serving or generation, copy `best.pt`:

```bash
mkdir -p checkpoints/latest
cp checkpoints/best/model.pt checkpoints/latest/model.pt
```

### Two-stage CPU training

Start a fresh general-language checkpoint on TinyStories and WikiText. This is
a large CPU job and can take a long time:

```bash
python scripts/train.py --model-config configs/model.cpu.yaml \
  --training-config configs/pretraining.cpu.yaml \
  --output checkpoints/pretraining/latest.pt \
  --best-output checkpoints/pretraining/best.pt
```

Initialize a new optimizer and fine-tune the best pretrained weights on
UltraChat, HelpSteer, and OpenOrca:

```bash
python scripts/train.py --model-config configs/model.cpu.yaml \
  --training-config configs/finetuning.cpu.yaml \
  --init-from checkpoints/pretraining/best.pt \
  --output checkpoints/finetuning/latest.pt \
  --best-output checkpoints/finetuning/best.pt
```

To use the best fine-tuned model for default serving and generation, copy the best checkpoint to `checkpoints/latest/model.pt`:

```bash
mkdir -p checkpoints/latest
cp checkpoints/finetuning/best.pt checkpoints/latest/model.pt
```

Use `--resume` only to continue the same training stage with its optimizer,
scheduler, sampler, random-number, and early-stopping state. Use `--init-from`
to transfer model weights into a new stage with a fresh optimizer and schedule.

For a CUDA GPU, use the full model with the conservative FP16 profiles:

```bash
python scripts/train.py --model-config configs/model.gpu.yaml \
  --training-config configs/pretraining.gpu.yaml \
  --output checkpoints/pretraining-gpu/latest.pt \
  --best-output checkpoints/pretraining-gpu/best.pt

python scripts/train.py --model-config configs/model.gpu.yaml \
  --training-config configs/finetuning.gpu.yaml \
  --init-from checkpoints/pretraining-gpu/best.pt \
  --output checkpoints/finetuning-gpu/latest.pt \
  --best-output checkpoints/finetuning-gpu/best.pt

mkdir -p checkpoints/latest
cp checkpoints/finetuning-gpu/best.pt checkpoints/latest/model.pt
```

These profiles use a micro-batch of two 512-token sequences and accumulate 16
micro-batches for an effective batch size of 32. Reduce `batch_size` to one if
CUDA runs out of memory. On GPUs with native BF16 support, changing
`mixed_precision` from `fp16` to `bf16` usually improves numerical robustness.

For a broader CPU experiment using every configured dataset, use:

```bash
python scripts/train.py --model-config configs/model.cpu.yaml \
  --training-config configs/training.cpu.yaml --epochs 1
```

Run the test suite with:

```bash
pytest -q
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
