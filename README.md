## Omni multimodal platform v5

See [`docs/OMNI_PLATFORM_V5.md`](docs/OMNI_PLATFORM_V5.md) for the production multimodal provider, routing, job, artifact, queue, capability, resource, and deployment architecture.

# Gopi LLM Engine

A configuration-driven language and media-model engine implemented with
PyTorch. It includes tokenizer training and safe vocabulary extension,
pretraining, supervised fine-tuning, recovery SFT, DPO, evaluation, export,
RAG, web search, image diffusion, audio/video latent diffusion, an
OpenAI-compatible API, and distributed-training building blocks.

This is an educational and experimental engine. It does not ship pretrained
weights, and its small-model profiles are not substitutes for production-scale
foundation models.

## Documentation

- [Project index](docs/PROJECT_INDEX.md) — authoritative source, config, and test locations.
- [Model reference](docs/MODEL.md) and [configuration reference](docs/CONFIGURATION.md).
- [Training guide](docs/TRAINING_GUIDE.md) and [model-growth guide](docs/DIRECT_TRAINING_GUIDE.md).
- [Usage guide](docs/USAGE_GUIDE.md), [deployment guide](docs/DEPLOYMENT.md), and [dataset catalog](docs/DATASET_CATALOG.md).
- [Capabilities and scaling](docs/CAPABILITIES_AND_SCALING.md), [vision and image diffusion](docs/IMAGE_MODELS.md), and [audio/video diffusion training](docs/TRAINING_GUIDE.md#11-audio-and-video-diffusion-training).

Documentation and active configuration filenames are unversioned.

## Current configuration layout

Active profiles, including the tokenizer, use unversioned filenames.

| Configuration | Purpose |
| --- | --- |
| `configs/model.gpu.yaml` | Active 40K-base, 16-layer, approximately 81.3M GPU model |
| `configs/model.cpu.yaml` | Smaller 32K, 8-layer CPU model |
| `configs/model.source.gpu.yaml` | Frozen 32K/10-layer shape for checkpoint growth |
| `configs/pretraining.gpu.yaml` | GPU continued-pretraining stage |
| `configs/pretraining.cpu.yaml` | CPU pretraining stage |
| `configs/finetuning.gpu.yaml` | Expanded multilingual GPU SFT stage |
| `configs/finetuning.cpu.yaml` | Quality-balanced CPU SFT stage |
| `configs/finetuning.recovery.gpu.yaml` | Focused response-quality recovery |
| `configs/dpo.gpu.yaml` / `dpo.cpu.yaml` | Preference optimization |
| `configs/inference.yaml` | CLI and serving defaults |
| `configs/tokenizer.yaml` | 40K base tokenizer with a 2K fine-tuning extension |
| `configs/evaluation.*` | Domain and fixed-case evaluation |
| `configs/*packed*` | Memory-mapped token-shard training |
| `configs/vision/multimodal.yaml` | Small multimodal adapter profile |
| `configs/diffusion/` | Pixel-space image diffusion and planning-only native latent-image profiles |
| `configs/audio_generation/` | Local, high-quality, and ultra-quality text-to-audio diffusion profiles |
| `configs/video_generation/` | Local, high-quality, and ultra-quality text/image/video-to-video diffusion profiles |
| `configs/text/` | Future 50K tokenizer and 1B/7B/30B targets |
| `configs/scaling/model.moe-100b.yaml` | Sparse 97.28B planning profile with about 13.96B parameters active per token |

## Environment setup

Python 3.10 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --editable '.[dev]'
python scripts/capabilities.py
```

Optional dependency groups are available for ONNX export, PDF RAG, and image
support. Install every runtime feature with `python -m pip install -e '.[full]'`,
or a development environment with all features using
`python -m pip install -e '.[dev,full]'`.

## Quick start

The active GPU model uses the 40K base tokenizer written to `data/tokenizer`:

```bash
.venv/bin/python scripts/tokenize.py train --config configs/tokenizer.yaml
```

Before starting the active SFT profile, build its append-only tokenizer extension:

```bash
.venv/bin/python scripts/tokenize.py extend --config configs/tokenizer.yaml --extension finetuning
```

For a fresh training run:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/python scripts/train.py \
  --model-config configs/model.gpu.yaml \
  --training-config configs/pretraining.gpu.yaml \
  --tokenizer data/tokenizer \
  --output checkpoints/pretraining/latest.pt \
  --best-output checkpoints/pretraining/best.pt
```

Start SFT as a new optimizer stage:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/python scripts/train.py \
  --model-config configs/model.gpu.yaml \
  --training-config configs/finetuning.gpu.yaml \
  --tokenizer data/tokenizer-finetuning \
  --init-from checkpoints/pretraining/best.pt \
  --output checkpoints/finetuning/latest.pt \
  --best-output checkpoints/finetuning/best.pt
```

Use `--resume checkpoints/finetuning/latest.pt` only to continue that same
stage. `--init-from` loads weights into a new stage with fresh optimizer,
scheduler, sampler, and report history.

The fine-tuning profile also writes `checkpoints/finetuning/best-generation.pt`
when fixed-prompt answer accuracy improves. Ties retain the earlier checkpoint;
`best.pt` still tracks validation loss. The generation checkpoint contains the
exact evaluated weights (EMA when enabled) and is for inference only. Continue
training from `latest.pt`. After the generation checkpoint has been created,
pass it as `--checkpoint` to chat/evaluation, or set
`serving.checkpoint_path` in `configs/inference.yaml` to use it for serving.
Per-check answers are retained in `reports/generation_quality_history/`.
Changing the evaluation prompts, decoding settings, or tokenizer requires a new
`generation_evaluation.best_output` path so unlike scores are not compared.
The 18-prompt suite is a small diagnostic; its best score does not establish
broad model quality or guarantee that every individual answer is retained.

## Training reports

Rank zero launches an isolated live-report process. It reads
`logs/training.log` and atomically refreshes
`reports/training_report.json` without loading the model or using GPU memory.

- New runs and `--init-from` archive prior log/JSON files with timestamped
  `.previous-*` names and start a clean dashboard.
- `--resume` preserves and appends the interrupted run's history.

Serve the static dashboard from another terminal:

```bash
.venv/bin/python -m http.server 8000 --directory reports
```

Open `http://localhost:8000/training_report.html`. The dashboard automatically selects the newest training or fine-tuning report. To follow fine-tuning explicitly, use `http://localhost:8000/training_report.html?data=finetuning.json`. Dataset weights take effect when training starts or resumes; editing the YAML does not change a running trainer.

## Generate and chat

```bash
.venv/bin/python scripts/generate.py "Hello Gopi" \
  --model-config configs/model.gpu.yaml \
  --inference-config configs/inference.yaml \
  --tokenizer data/tokenizer-finetuning \
  --checkpoint checkpoints/finetuning/best.pt \
  --device cuda
```

```bash
.venv/bin/python scripts/chat.py \
  --model-config configs/model.gpu.yaml \
  --inference-config configs/inference.yaml \
  --tokenizer data/tokenizer-finetuning \
  --checkpoint checkpoints/finetuning/best.pt \
  --device cuda
```

## Evaluate

```bash
.venv/bin/python scripts/evaluate_domains.py \
  --domains configs/evaluation.finetuning.yaml \
  --model-config configs/model.gpu.yaml \
  --tokenizer data/tokenizer-finetuning \
  --checkpoint checkpoints/finetuning/best.pt \
  --device cuda

.venv/bin/python scripts/evaluate_benchmarks.py \
  --cases configs/evaluation.domains.jsonl \
  --model-config configs/model.gpu.yaml \
  --inference-config configs/inference.yaml \
  --tokenizer data/tokenizer-finetuning \
  --checkpoint checkpoints/finetuning/best.pt \
  --device cuda
```

Select deployable checkpoints using held-out validation and fixed behavioral
evaluation, not training loss alone.

Persist the two evaluations to make them available in the live report:

```bash
.venv/bin/python scripts/audit_data_quality.py \
  --training-config configs/finetuning.gpu.yaml \
  --output reports/data_quality.json

.venv/bin/python scripts/evaluate_benchmarks.py \
  --cases configs/evaluation.domains.jsonl \
  --model-config configs/model.gpu.yaml \
  --inference-config configs/inference.yaml \
  --tokenizer data/tokenizer-finetuning \
  --checkpoint checkpoints/finetuning/best.pt \
  --device cuda \
  --output reports/generation_quality.json
```

## DPO and export

```bash
.venv/bin/python scripts/train_dpo.py \
  --model-config configs/model.gpu.yaml \
  --training-config configs/dpo.gpu.yaml \
  --tokenizer data/tokenizer-finetuning \
  --reference-checkpoint checkpoints/finetuning/best.pt \
  --init-from checkpoints/finetuning/best.pt \
  --output checkpoints/dpo/latest.pt \
  --best-output checkpoints/dpo/best.pt \
  --device cuda
```

```bash
.venv/bin/python scripts/export.py \
  --model-config configs/model.gpu.yaml \
  --tokenizer data/tokenizer-finetuning \
  --checkpoint checkpoints/dpo/best.pt \
  --format safetensors \
  --output exports/final/gopi.safetensors
```

## Serving

`configs/inference.yaml` defaults to model name `gopi`,
`configs/model.gpu.yaml`, `data/tokenizer-finetuning`, and
`checkpoints/finetuning/best.pt`.

```bash
export GOPI_API_KEY='replace-with-a-long-random-value'
.venv/bin/python scripts/serve.py --host 127.0.0.1 --port 8000
```

Verify liveness and readiness:

```bash
curl --fail http://127.0.0.1:8000/health/live
curl --fail http://127.0.0.1:8000/health/ready
```

The server provides OpenAI-compatible HTTP endpoints, SSE and WebSocket
streaming, bounded concurrency, prefix caching, paged KV-cache accounting,
session/rate-limit stores, optional RAG, optional web search, and MCP client
integration. Treat retrieved content as untrusted.

For containers:

```bash
cp .env.production.example .env.production
# Replace GOPI_API_KEY before continuing.
docker compose --env-file .env.production up --build -d
```

Review the [deployment guide](docs/DEPLOYMENT.md) before exposing the service publicly.

## Architecture and training behavior

The model is a decoder-only causal transformer with byte-level BPE,
Unicode-aware pre-tokenization, GQA, RoPE, RMSNorm, SwiGLU, tied embeddings,
gradient accumulation, mixed precision, EMA, deterministic checkpoints, DDP,
optional FSDP, and distributed checkpoint support.

Checkpoints record tokenizer fingerprints. Training and inference reject a
different same-size tokenizer unless it is a verified append-only descendant.
Growing a checkpoint uses `configs/model.source.gpu.yaml`,
`configs/model.gpu.yaml`, and `scripts/grow_checkpoint.py`.

## Dataset governance

Processed datasets have manifests describing provenance, license review,
privacy review, and allowed stages. Audit exact inputs before a run:

```bash
python scripts/audit_datasets.py \
  --training-config configs/finetuning.gpu.yaml --stage sft
```

`warn` reports findings, `error` blocks training, and `off` explicitly disables
the check. A manifest records review; it does not replace legal or privacy
assessment.

### Clean and pack pretraining JSONL

The preparation tools produce one JSONL file per split, never binary token
shards. Clean validation first, then exclude it while cleaning training data so
train/validation overlap is removed:

```bash
.venv/bin/python scripts/clean_jsonl_corpus.py \
  data/processed/wikitext_103/validation.jsonl \
  --output data/cleaned/wikitext_103/validation.jsonl \
  --tokenizer data/tokenizer

.venv/bin/python scripts/clean_jsonl_corpus.py \
  data/processed/wikitext_103/train.jsonl \
  --output data/cleaned/wikitext_103/train.jsonl \
  --tokenizer data/tokenizer \
  --exclude data/cleaned/wikitext_103/validation.jsonl

.venv/bin/python scripts/pack_jsonl_corpus.py \
  data/cleaned/wikitext_103/train.jsonl \
  --output data/cleaned/wikitext_103/train.packed.jsonl \
  --tokenizer data/tokenizer --sequence-length 512
```

Repeat this workflow for TinyStories. The cleaner normalizes Unicode, filters
low-quality and duplicate text, redacts common secrets and personal identifiers,
detects language, measures token lengths, and writes an `.audit.json` report.
The packer combines short documents with EOS boundaries and writes a
`.packing.json` report. Generated files under `data/cleaned/` are ignored; the
empty directory is retained with `.gitkeep`.

`configs/pretraining.cleaned.gpu.yaml` is a separate new-stage configuration.
It is intentionally unusable until all referenced packed files have been
generated and audited; do not use it to resume a sampler created from
`configs/pretraining.gpu.yaml`.

### Additional local causal-LM data

The workspace includes two bounded, optional JSONL corpora for a later broad
pretraining stage:

- `data/processed/fineweb_edu/`: 533,797 train, 2,000 validation, and 2,011
  test educational-web records (about 2.5 GiB processed). The source is the
  FineWeb-Edu `sample-10BT` configuration under ODC-By-1.0 and Common Crawl
  terms; attribution is required and privacy review is incomplete.
- `data/processed/code_pretraining/`: 79,822 train, 1,000 validation, and
  1,000 test raw Python-code records (about 803 MiB processed). Source files
  carry mixed per-record licenses, so license and privacy review are incomplete.

These datasets are not included in `configs/pretraining.gpu.yaml`. Keep the
active 90% WikiText / 10% TinyStories run unchanged when resuming its existing
checkpoint. The GPU fine-tuning profile uses each corpus at 2% for causal-LM
retention and `configs/tokenizer.yaml` uses them to build the append-only
`data/tokenizer-finetuning` artifact. Start this as a new fine-tuning stage; do
not resume an older sampler with the changed mixture.

## Future scale targets

`configs/text/` contains separate, intentionally incompatible future profiles:

- 50K from-scratch multilingual tokenizer;
- approximately 1.185B model with an 8K context;
- 7B and 30B cluster targets with 32K contexts;
- an opt-in FSDP pretraining example.

These are architecture targets, not laptop training recommendations. Use
`scripts/inspect_model.py` and `scripts/plan_training.py` before allocating
hardware.

### Configuration-driven sparse MoE

Transformer FFNs can be changed from dense layers to sparse Mixture-of-Experts
layers entirely through the model YAML:

```yaml
ffn_type: moe
ffn_hidden_size: 11008
ffn_activation: swiglu
num_experts: 16
experts_per_token: 2
router_bias: false
router_jitter: 0.01
```

`num_experts` controls stored expert capacity. `experts_per_token` controls how
many experts execute for each token. `router_jitter` adds multiplicative router
input noise during training only; it is disabled automatically during
evaluation and generation. Use `ffn_type: dense` (the default) for existing
models and checkpoints. Changing between dense and MoE changes checkpoint
shapes and requires training a compatible model.

Inspect the supplied 100B-class template without allocating its weights:

```bash
.venv/bin/python scripts/inspect_model.py configs/scaling/model.moe-100b.yaml
```

The template stores approximately 97.28B parameters and routes each token
through approximately 13.96B active parameters. It deliberately keeps
`planning_only: true`: all expert weights require about 181.2 GiB in BF16, and
this engine does not yet implement expert-parallel sharded checkpoint loading.
Small MoE profiles can use the normal training and generation commands.
Tensor-parallel MoE serving is rejected explicitly until expert parallelism is
available; changing only the YAML cannot overcome physical weight memory.

### Lower-memory inference

Serving, `scripts/generate.py`, and `scripts/chat.py` read these options from
the `serving` section of `configs/inference.yaml`:

```yaml
serving:
  low_memory_loading: true
  weight_dtype: bfloat16       # float32, float16, or bfloat16
  quantization: none           # none or int8_dynamic
```

`low_memory_loading` memory-maps a PyTorch checkpoint and assigns tensors
without first creating a second in-memory state-dict copy. `bfloat16` roughly
halves resident weight memory relative to FP32. For CPU-only inference,
`quantization: int8_dynamic` compresses Linear weights further; CUDA serving
must use `quantization: none`. CPU FP16 is rejected because its kernel support
is unsuitable here; use BF16 or INT8 instead.

These controls lower loading peaks and resident memory, but they do not turn a
100B model into an 8–16 GiB model. The supplied MoE profile still needs about
181.2 GiB for BF16 weights or roughly 91 GiB at INT8 before runtime overhead.
Disk-backed expert paging and expert-parallel checkpoint shards remain future
work; the 100B profile therefore remains planning-only.

## Test suite

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/audit_task_registry.py
```

`pyarrow` is a declared core dependency used by the Arrow/Hugging Face data-preparation tests. In an offline/source-only environment where it is not installed, report those modules as not executed rather than treating them as passing. Generated tokenizer artifacts and generated long-context/reasoning-SFT fixtures are intentionally gitignored, so their artifact-dependent tests skip when absent from a source archive.

## Audio and video generation

The repository includes checkpoint-backed diffusion stacks for text-to-audio,
audio-to-audio, long-form audio, text-to-video, image-to-video, and
video-to-video generation. Supported workflows include multi-segment extension,
negative prompts, CFG, named inference presets, batch generation, reproducibility
manifests, and guarded HTTP model serving. These are separate checkpoint
families from the language model; see the [training guide](docs/TRAINING_GUIDE.md#11-audio-and-video-diffusion-training),
[model formulation](docs/MODEL.md#6-audio-and-video-latent-diffusion-models),
and [configuration reference](docs/CONFIGURATION.md#6-audio-and-video-generation-configuration).
