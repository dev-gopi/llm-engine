# Gopi LLM Engine

Gopi is an educational, configuration-driven project for building a GPT-style
language model from the ground up. The repository separates tokenization, data
processing, model architecture, training, inference, and serving so each part
can evolve independently as the model grows.

> **Development status:** This repository currently provides the project
> structure, starter datasets, and a developing Transformer implementation. Several
> training, inference, and checkpointing modules are still placeholders.
> It is not production-ready yet.

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

Architecture values live in [`configs/model.yaml`](configs/model.yaml), keeping
the model implementation independent from experiment size.

### Token embedding matrix

`TokenEmbedding` maps vocabulary IDs into the model's hidden dimension. It uses
GPT-style normal initialization, keeps the padding row zero, optionally scales
vectors by the square root of the hidden size, and supports freezing, safe
vocabulary resizing, hardware-aligned vocabulary padding, and weight sharing
with the language-model output projection. Device and floating-point dtype can
be selected at construction time.

With the default configuration, the matrix contains `50,000 × 768 = 38.4M`
parameters. Its initialization and behavior are controlled by
`padding_idx`, `initializer_range`, `scale_embeddings`, and
`freeze_embeddings` in [`configs/model.yaml`](configs/model.yaml).

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
are configured in [`configs/model.yaml`](configs/model.yaml). A final LayerNorm
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

Serving defaults come from [`configs/inference.yaml`](configs/inference.yaml)
and can be overridden with `GOPI_MODEL_NAME`, `GOPI_BOT_NAME`,
`GOPI_MAX_CONCURRENCY`, `GOPI_QUEUE_TIMEOUT_SECONDS`,
`GOPI_GENERATION_TIMEOUT_SECONDS`, and `GOPI_CORS_ORIGINS`.

The default application intentionally reports `503 not_ready` until a concrete
asynchronous generation backend is injected. This prevents a healthy-looking
deployment from accepting traffic before model weights and tokenizer artifacts
are loaded. Authentication, TLS, and global rate limiting should be enforced by
the deployment gateway or ingress layer.

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
```

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

| File | Responsibility |
| --- | --- |
| `configs/model.yaml` | Vocabulary size and Transformer architecture |
| `configs/training.yaml` | Batch size, epochs, learning rate, and weight decay |
| `configs/tokenizer.yaml` | Tokenizer type and vocabulary size |
| `configs/inference.yaml` | Gopi's identity and generation behavior |

Training code must not read inference settings, and inference code must not
depend on the training package.

## Commands

The intended command-line workflows are:

```bash
python scripts/tokenize.py train
python scripts/train.py
python scripts/evaluate.py
python scripts/generate.py
python scripts/export.py
python scripts/serve.py
```

These entrypoints currently remain scaffolds and will become functional as their
respective subsystems are implemented.

Run the test suite with:

```bash
pytest -q
```

## Current implementation status

| Area | Status |
| --- | --- |
| Project architecture and configuration | Scaffolded |
| Dataset acquisition and chat preprocessing | Working |
| Token embedding matrix | Production implementation |
| Positional embeddings | Minimal implementation |
| Transformer blocks and language-model head | Minimal implementation |
| Causal QKV self-attention and attention masks | Production implementation |
| Position-wise FFN with GELU/SwiGLU support | Production implementation |
| Pre-norm residual connections and LayerNorm/RMSNorm | Production implementation |
| Shifted causal LM loss and perplexity | Production implementation |
| Byte-level BPE tokenizer and artifact persistence | Working |
| Full training and evaluation loops | Not implemented |
| Checkpoint save and resume | Not implemented |
| Autoregressive generation and sampling | Not implemented |
| Per-layer attention KV cache | Working |
| REST/WebSocket serving infrastructure | Production implementation |
| Model-backed serving generation | Waiting for inference engine |
| Model export | Not implemented |
| Meaningful unit tests | Working and expanding |

## Roadmap

1. Build dataset loaders and causal language-model collators.
2. Complete positional encoding and language-model head configuration.
3. Implement training, validation, scheduling, and checkpoint recovery.
4. Add autoregressive generation with top-k/top-p sampling.
5. Connect per-layer KV caches across the complete model and stream tokens.
6. Connect the inference generator backend to the serving runtime.
7. Add SafeTensors, ONNX, and GGUF export workflows.
8. Expand correctness, reproducibility, and integration tests.
9. Add distributed training only after the single-device pipeline is stable.

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
