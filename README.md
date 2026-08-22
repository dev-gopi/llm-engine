# Gopi LLM Engine

Gopi is an educational, configuration-driven project for building a GPT-style
language model from the ground up. The repository separates tokenization, data
processing, model architecture, training, inference, and serving so each part
can evolve independently as the model grows.

> **Development status:** The complete local pipeline is implemented: dataset
> preparation, tokenization, GPT training and resume, evaluation, cached
> generation, model-backed serving, and model export. Production deployment
> still requires appropriately licensed data, trained weights, capacity testing,
> security controls, and operational monitoring.

## Goals

- Learn and implement the complete LLM pipeline rather than wrapping a hosted model.
- Keep experiments reproducible through YAML configuration.
- Maintain clear boundaries between training and inference code.
- Scale from a small local model toward distributed training and optimized serving.
- Support checkpoint recovery and multiple deployment formats.

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
│   ├── inference/           # Generation, sampling, and KV caching
│   ├── serving/             # FastAPI and WebSocket interfaces
│   └── utils/               # Configuration, devices, logging, and seeds
├── checkpoints/             # Resumable training state
├── exports/                 # SafeTensors, ONNX, and GGUF artifacts
├── scripts/                 # Command-line workflows
└── tests/                   # Unit and integration tests
```

## Model architecture

The current model is a compact GPT-style network composed of:

1. Learned token embeddings
2. Learned positional embeddings
3. Stacked Transformer blocks
4. Multi-head self-attention
5. GELU feed-forward networks
6. Layer normalization and residual connections
7. A vocabulary projection head

Architecture values live in model configurations like [`configs/model.gpu.yaml`](configs/model.gpu.yaml) (full model) or [`configs/model.cpu.yaml`](configs/model.cpu.yaml) (compact model), keeping
the model implementation independent from experiment size.

### Token embedding matrix

`TokenEmbedding` maps vocabulary IDs into the model's hidden dimension. It uses
GPT-style normal initialization, keeps the padding row zero, optionally scales
vectors by the square root of the hidden size, and supports freezing, safe
vocabulary resizing, hardware-aligned vocabulary padding, and weight sharing
with the language-model output projection. Device and floating-point dtype can
be selected at construction time.

With the GPU configuration (`configs/model.gpu.yaml`), the matrix contains `50,000 × 768 = 38.4M`
parameters (or `50,000 × 128 = 6.4M` in `configs/model.cpu.yaml`). Its initialization and behavior are controlled by
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
token position. The default configuration expands each 768-dimensional token
to 3,072 dimensions, applies GELU, projects back to the model dimension, and
then applies dropout.

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

At startup the application loads the configured tokenizer, model configuration,
and checkpoint. It reports `503 not_ready` when an artifact is absent or the
backend cannot be loaded. Authentication, TLS, and global rate limiting should
be enforced by the deployment gateway or ingress layer.

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

The generated train and validation splits are included in training configuration profiles (such as `configs/pretraining.cpu.yaml`, `configs/pretraining.gpu.yaml`, `configs/finetuning.cpu.yaml`, `configs/finetuning.gpu.yaml`, `configs/training.cpu.yaml`, and `configs/training.gpu.yaml`) alongside the conversational datasets.

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

### DailyDialog subset

DailyDialog adds human-written, everyday conversations. The processed subset
contains 2,000 user/assistant pairs.

- License: CC BY-NC-SA 4.0
- Source: [ConvLab/dailydialog](https://huggingface.co/datasets/ConvLab/dailydialog)
- Commercial-use warning: this dataset is non-commercial and share-alike.

Regenerate it with:

```bash
.venv/bin/python scripts/prepare_dailydialog.py
.venv/bin/python scripts/split_dailydialog.py
```

The second command creates deterministic `train.jsonl` and `validation.jsonl`
files while keeping every pair from one source dialogue in the same split. The
current 2,000-pair subset produces 1,803 training and 197 validation records.

Raw data and generated training artifacts are intentionally excluded from Git.
Dataset-specific provenance is documented under `data/processed/<dataset>/README.md`.

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
| `configs/model.cpu.yaml` | Compact model for CPU development and smoke tests | `hidden_size: 128`, `layers: 4`, `heads: 4`, `max_position: 512`, `ffn_hidden_size: 512` |
| `configs/model.gpu.yaml` | Full model configuration for GPU training and production | `hidden_size: 768`, `layers: 12`, `heads: 12`, `max_position: 2048`, `ffn_hidden_size: 3072` |
| `configs/pretraining.cpu.yaml` | Pretraining configuration for CPU | DailyDialog dataset, `batch_size: 2`, `max_sequence_length: 256`, `gradient_accumulation_steps: 4`, 1 epoch |
| `configs/pretraining.gpu.yaml` | Pretraining configuration for GPU | DailyDialog dataset, `batch_size: 2`, `max_sequence_length: 512`, `gradient_accumulation_steps: 16`, `mixed_precision: fp16`, 3 epochs |
| `configs/finetuning.cpu.yaml` | Supervised fine-tuning configuration for CPU | WikiText + UltraChat datasets, `batch_size: 2`, `max_sequence_length: 256`, `gradient_accumulation_steps: 4`, 3 epochs |
| `configs/finetuning.gpu.yaml` | Supervised fine-tuning configuration for GPU | WikiText + UltraChat datasets, `batch_size: 2`, `max_sequence_length: 512`, `gradient_accumulation_steps: 16`, `mixed_precision: fp16`, 3 epochs |
| `configs/training.cpu.yaml` | Full CPU profile across all datasets | WikiText + UltraChat + DailyDialog, `batch_size: 2`, `max_sequence_length: 256`, `gradient_accumulation_steps: 4`, 1 epoch |
| `configs/training.gpu.yaml` | Full GPU profile across all datasets | WikiText + UltraChat + DailyDialog, `batch_size: 32`, `max_sequence_length: 2048`, `gradient_accumulation_steps: 1`, 10 epochs |
| `configs/tokenizer.yaml` | Byte-level BPE tokenizer training setup | `vocab_size: 50000`, `min_frequency: 2`, `special_tokens`, sources list |
| `configs/inference.yaml` | Assistant identity, sampling, and API serving defaults | `bot_name: Gopi`, temperature, top_k/p, context_memory, sqlite session path, concurrency & rate limits |

Training code must not read inference settings, and inference code must not
depend on the training package.

## Commands

The command-line workflows are:

```bash
python scripts/tokenize.py train
python scripts/train.py
python scripts/evaluate.py
python scripts/generate.py "Hello Gopi"
python scripts/export.py --format safetensors
python scripts/serve.py
```

Training settings, dataset paths, checkpoint frequency, optimizer, scheduler,
and EMA behavior are controlled by configuration files such as `configs/training.cpu.yaml` or `configs/training.gpu.yaml`. Training resumes
with `python scripts/train.py --resume checkpoints/latest/model.pt`.
`mixed_precision` accepts `none`, `bf16`, or CUDA-only `fp16`, while
`gradient_accumulation_steps` controls effective batch size.

### First small training run

Before starting the complete corpus, verify the pipeline with the compact CPU
model and the 2,000-pair DailyDialog subset using `configs/pretraining.cpu.yaml`:

```bash
python scripts/train.py --model-config configs/model.cpu.yaml \
  --training-config configs/pretraining.cpu.yaml --epochs 1
```

This profile uses two examples per batch, 256-token sequences, four-step
gradient accumulation, and a single-process data loader. It writes the final
checkpoint to `checkpoints/latest/model.pt`. On the current reference run, one
epoch completed 250 optimizer steps and reduced the running training loss from
approximately 10.57 to 6.75.

Evaluate a bounded sample without loading the full dataset into memory:

```bash
python scripts/evaluate.py --model-config configs/model.cpu.yaml \
  --training-config configs/pretraining.cpu.yaml \
  --dataset data/processed/dailydialog/dailydialog-conversations.json \
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

Start a fresh general-language checkpoint on WikiText. This is a large CPU job,
so expect it to take substantially longer than the DailyDialog smoke run:

```bash
python scripts/train.py --model-config configs/model.cpu.yaml \
  --training-config configs/pretraining.cpu.yaml \
  --output checkpoints/pretraining/latest.pt \
  --best-output checkpoints/pretraining/best.pt
```

Initialize a new optimizer and fine-tune the best pretrained weights on
UltraChat plus the leakage-resistant DailyDialog training split:

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
| Project architecture and configuration | Production implementation |
| Dataset acquisition and chat preprocessing | Production implementation |
| Token embedding matrix | Production implementation |
| Positional embeddings | Production implementation |
| Transformer blocks and language-model head | Production implementation |
| Causal QKV self-attention and attention masks | Production implementation |
| Position-wise FFN with GELU/SwiGLU support | Production implementation |
| Pre-norm residual connections and LayerNorm/RMSNorm | Production implementation |
| Shifted causal LM loss and perplexity | Production implementation |
| Byte-level BPE tokenizer and artifact persistence | Production implementation |
| Full training and evaluation loops | Production implementation |
| Checkpoint save and resume | Production implementation |
| Autoregressive generation and sampling | Production implementation |
| Per-layer attention KV cache | Production implementation |
| REST/WebSocket serving infrastructure | Production implementation |
| Model-backed serving generation | Production implementation |
| SafeTensors and PyTorch Export | Production implementation |
| ONNX export | Implemented; install `.[export]` and validate in the target runtime |
| Meaningful unit and integration tests | Production implementation |

## Roadmap

1. Train and publish reproducible small-model reference checkpoints.
2. Benchmark automatic mixed precision and accumulation on target GPUs.
3. Add sharded FSDP checkpointing for multi-node training.
4. Add a separately validated GGUF conversion workflow.
5. Add continuous request batching for higher serving throughput.
6. Run ONNX Runtime compatibility, deployment load, and failure-recovery tests.

## Design principles

- One responsibility per module
- Configuration instead of hardcoded experiments
- Immutable raw data and reproducible preprocessing
- Independent tokenizer and model artifacts
- No dependency from inference to training
- Versioned, resumable checkpoints
- Tests before performance optimization
- Honest documentation of unfinished functionality

## License

The project does not currently declare a repository-level software license.
Dataset licenses apply independently; review them before distributing data or
using trained artifacts commercially.
