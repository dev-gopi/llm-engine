# Gopi LLM Engine

A configuration-driven GPT-style language-model engine implemented with
PyTorch. It includes tokenizer training and safe vocabulary extension,
pretraining, supervised fine-tuning, recovery SFT, DPO, evaluation, export,
RAG, web search, an OpenAI-compatible API, and distributed-training building
blocks.

This is an educational and experimental engine. It does not ship pretrained
weights, and its small-model profiles are not substitutes for production-scale
foundation models.

## Guides

- [Training guide](docs/TRAINING_GUIDE.md) — prepare datasets and the
  tokenizer, pretrain, fine-tune, run recovery or DPO, evaluate, and export.
- [Direct model-growth guide](docs/DIRECT_TRAINING_GUIDE.md) — extend the
  tokenizer, grow a compatible source checkpoint, initialize a new stage,
  resume it safely, and monitor training.
- [Usage guide](docs/USAGE_GUIDE.md) — generate from the CLI, chat in the
  terminal, use the browser/API, stream responses, and run exports.
- [Capabilities and scaling guide](docs/CAPABILITIES_AND_SCALING.md) —
  realistic uses, limitations, and the path toward the provided 1B, 7B, and
  30B architecture targets.
- [Dataset catalog](docs/DATASET_CATALOG.md) — dataset purposes, provenance,
  preparation, and governance.
- [Deployment guide](docs/DEPLOYMENT.md) — local and container serving,
  security, monitoring, and recovery.
- [Vision and diffusion guide](docs/IMAGE_MODELS.md) — image datasets, training,
  checkpoint resume, classification, sampling, and production validation.

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
| `configs/text/` | Future 50K tokenizer and 1B/7B/30B targets |

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
  --tokenizer data/tokenizer \
  --init-from checkpoints/pretraining/best.pt \
  --output checkpoints/finetuning/latest.pt \
  --best-output checkpoints/finetuning/best.pt
```

Use `--resume checkpoints/finetuning/latest.pt` only to continue that same
stage. `--init-from` loads weights into a new stage with fresh optimizer,
scheduler, sampler, and report history.

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

Open `http://localhost:8000/training_report.html`.

## Generate and chat

```bash
.venv/bin/python scripts/generate.py "Hello Gopi" \
  --model-config configs/model.gpu.yaml \
  --inference-config configs/inference.yaml \
  --tokenizer data/tokenizer \
  --checkpoint checkpoints/finetuning/best.pt \
  --device cuda
```

```bash
.venv/bin/python scripts/chat.py \
  --model-config configs/model.gpu.yaml \
  --inference-config configs/inference.yaml \
  --tokenizer data/tokenizer \
  --checkpoint checkpoints/finetuning/best.pt \
  --device cuda
```

## Evaluate

```bash
.venv/bin/python scripts/evaluate_domains.py \
  --domains configs/evaluation.finetuning.yaml \
  --model-config configs/model.gpu.yaml \
  --tokenizer data/tokenizer \
  --checkpoint checkpoints/finetuning/best.pt \
  --device cuda

.venv/bin/python scripts/evaluate_benchmarks.py \
  --cases configs/evaluation.domains.jsonl \
  --model-config configs/model.gpu.yaml \
  --inference-config configs/inference.yaml \
  --tokenizer data/tokenizer \
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
  --tokenizer data/tokenizer \
  --checkpoint checkpoints/finetuning/best.pt \
  --device cuda \
  --output reports/generation_quality.json
```

## DPO and export

```bash
.venv/bin/python scripts/train_dpo.py \
  --model-config configs/model.gpu.yaml \
  --training-config configs/dpo.gpu.yaml \
  --tokenizer data/tokenizer \
  --reference-checkpoint checkpoints/finetuning/best.pt \
  --init-from checkpoints/finetuning/best.pt \
  --output checkpoints/dpo/latest.pt \
  --best-output checkpoints/dpo/best.pt \
  --device cuda
```

```bash
.venv/bin/python scripts/export.py \
  --model-config configs/model.gpu.yaml \
  --tokenizer data/tokenizer \
  --checkpoint checkpoints/dpo/best.pt \
  --format safetensors \
  --output exports/final/gopi.safetensors
```

## Serving

`configs/inference.yaml` defaults to model name `gopi`,
`configs/model.gpu.yaml`, `data/tokenizer`, and
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

Read [DEPLOYMENT.md](docs/DEPLOYMENT.md) before exposing the service publicly.

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

## Test suite

```bash
.venv/bin/python -m pytest -q
```
