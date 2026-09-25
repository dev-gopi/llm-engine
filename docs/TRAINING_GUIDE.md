# Model training guide

> The active model, tokenizer, and training profiles use unversioned filenames.

Run commands from the repository root. Review dataset licenses, privacy, and
manifests before training.

## 1. Install and inspect

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --editable '.[dev]'
python scripts/capabilities.py
```

The engine's data pipeline is imported as `datasets.*`. This is distinct from
the optional Hugging Face `datasets` dependency; running tests through the
project environment ensures the engine's `src/datasets` package is selected.

GPU training profiles require CUDA. `configs/finetuning.gpu.yaml` and
`configs/dpo.gpu.yaml` require BF16 and FP16 support respectively. Use the CPU
profiles when CUDA is unavailable.

## 2. Prepare and audit datasets

Dataset preparation commands and provenance are maintained in
[DATASET_CATALOG.md](DATASET_CATALOG.md). After preparation, audit the exact
stage inputs:

```bash
python scripts/audit_datasets.py \
  --training-config configs/pretraining.gpu.yaml --stage pretraining

python scripts/audit_datasets.py \
  --training-config configs/finetuning.gpu.yaml --stage sft
```

Do not train through missing, malformed, unreviewed, or incompatible data
without consciously accepting the configured governance policy.

## 3. Prepare the tokenizer

The active 40K-base model uses `data/tokenizer`. The tokenizer is trained from all
29 active GPU fine-tuning sources plus TinyStories and WikiText-103:

Its source list contains all 29 training inputs in
`configs/finetuning.gpu.yaml` and both pretraining corpora. Validation files
remain excluded to prevent held-out evaluation text from influencing tokenizer
construction.

```bash
.venv/bin/python scripts/tokenize.py train --config configs/tokenizer.yaml
```

Verify the artifact:

```bash
.venv/bin/python scripts/tokenize.py inspect \
  --tokenizer data/tokenizer \
  "Hello, नमस्ते, বাংলা" --add-bos --add-eos
```

This is a new 40K tokenizer lineage. Do not reuse checkpoints created with a
different tokenizer fingerprint.

## 4. Obtain the active model shape

The active GPU model is 40K/16-layer and should be trained from scratch with
the new tokenizer. Inspect it without allocating full weights:

```bash
python scripts/inspect_model.py configs/model.gpu.yaml
```

## 5. Pretrain

For a fresh run, omit both `--init-from` and `--resume`:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/python scripts/train.py \
  --model-config configs/model.gpu.yaml \
  --training-config configs/pretraining.gpu.yaml \
  --tokenizer data/tokenizer \
  --output checkpoints/pretraining/latest.pt \
  --best-output checkpoints/pretraining/best.pt
```

To continue from model weights while starting a new optimizer stage, add
`--init-from PATH`. To resume an interrupted pretraining run, use:

```bash
.venv/bin/python scripts/train.py \
  --model-config configs/model.gpu.yaml \
  --training-config configs/pretraining.gpu.yaml \
  --tokenizer data/tokenizer \
  --resume checkpoints/pretraining/latest.pt \
  --output checkpoints/pretraining/latest.pt \
  --best-output checkpoints/pretraining/best.pt
```

## 6. Supervised fine-tuning

Fine-tuning is a new stage, so use `--init-from`:

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

Use `--resume checkpoints/finetuning/latest.pt` only when restarting this same
stage. New stages archive the previous live report; resumed stages retain it.

## 7. Optional recovery stage

Generate the focused dataset, then initialize recovery from the best SFT model:

```bash
.venv/bin/python scripts/prepare_recovery_sft.py \
  --output data/processed/recovery_sft

.venv/bin/python scripts/train.py \
  --model-config configs/model.gpu.yaml \
  --training-config configs/finetuning.recovery.gpu.yaml \
  --tokenizer data/tokenizer \
  --init-from checkpoints/finetuning/best.pt \
  --output checkpoints/recovery/latest.pt \
  --best-output checkpoints/recovery/best.pt
```

Recovery targets response quality; it does not replace broad pretraining.

## 8. DPO

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

Use `configs/dpo.cpu.yaml` and `--device cpu` for the CPU route.

## 9. Evaluate

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
  --checkpoint checkpoints/dpo/best.pt \
  --device cuda
```

## 10. Export

```bash
.venv/bin/python scripts/export.py \
  --model-config configs/model.gpu.yaml \
  --tokenizer data/tokenizer \
  --checkpoint checkpoints/dpo/best.pt \
  --format safetensors \
  --output exports/final/gopi.safetensors
```

Keep `model.yaml` and the copied tokenizer beside the exported weights.

## 11. Audio and video diffusion training

Audio and video diffusion models are distinct from the language-model
checkpoint lineage. They require their own captioned datasets, configurations,
and checkpoints. Install the optional media dependencies before preparing data
or writing generated media:

```bash
.venv/bin/python -m pip install --editable '.[dev,media]'
```

Use one media file and one UTF-8 caption sidecar per example. The manifest
builder recognizes `.wav` for audio and `.mp4`, `.mov`, `.mkv`, or `.webm` for
video. For example:

```text
data/source_audio/rain.wav
data/source_audio/rain.txt
data/source_video/fox.mp4
data/source_video/fox.txt
```

Each non-empty caption sidecar describes its adjacent media file. Build a
deterministic train/validation split and inspect the JSONL output before
training:

```bash
.venv/bin/python scripts/prepare_media_manifest.py \
  --kind audio --root data/source_audio --output-dir data/audio

.venv/bin/python scripts/prepare_media_manifest.py \
  --kind video --root data/source_video --output-dir data/video
```

The resulting records have the following contracts; paths may be absolute or
relative to the manifest directory:

```json
{"audio": "/absolute/path/rain.wav", "text": "Steady rain on a window"}
{"video": "/absolute/path/fox.mp4", "text": "A red fox walking through snow"}
```

### 11.1 Low-memory profiles

[`configs/audio_generation/local_4gb.yaml`](../configs/audio_generation/local_4gb.yaml)
trains on 16 kHz, four-second mono clips with batch size 1, 16 accumulation
steps, FP16, and gradient clipping. [`configs/video_generation/local_4gb.yaml`](../configs/video_generation/local_4gb.yaml)
trains on 8-frame, 64×64 RGB clips at 8 FPS with the same memory-oriented batch
and accumulation settings. These are development-scale profiles, not a route to
high-resolution production video on a 4 GB GPU.

Use the media tokenizer specified by the selected configuration. A checkpoint
can only be resumed with the same architecture and tokenizer fingerprint.

### 11.2 Train or resume

Start a fresh audio or video run with the relevant local profile:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/python scripts/train_audio_generation.py \
  --config configs/audio_generation/local_4gb.yaml --device cuda

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/python scripts/train_video_generation.py \
  --config configs/video_generation/local_4gb.yaml --device cuda
```

The profiles write rolling checkpoints to `checkpoints/audio_generation/latest.pt`
and `checkpoints/video_generation/latest.pt`. When a validation manifest is
configured, the EMA-weighted lowest-validation-loss checkpoint is also written
to the corresponding `best.pt` path. Resume an interrupted run with its
rolling checkpoint:

```bash
.venv/bin/python scripts/train_audio_generation.py \
  --config configs/audio_generation/local_4gb.yaml \
  --resume checkpoints/audio_generation/latest.pt --device cuda

.venv/bin/python scripts/train_video_generation.py \
  --config configs/video_generation/local_4gb.yaml \
  --resume checkpoints/video_generation/latest.pt --device cuda
```

The trainers restore model, optimizer, scheduler, EMA, AMP scaler, progress,
and RNG state. They fail early for missing manifest files, invalid media shapes,
or incompatible tokenizer/checkpoint metadata.

### 11.3 Validate and sample

Run the focused contract tests before a long job:

```bash
.venv/bin/python -m pytest tests/test_audio_video_generation.py \
  tests/test_media_generation_production.py -q
```

After training, generate a small verification sample from the best video
checkpoint:

```bash
.venv/bin/python scripts/generate_video.py \
  --config configs/video_generation/local_4gb.yaml \
  --checkpoint checkpoints/video_generation/best.pt \
  --prompt "A red fox walking through snowy woods" \
  --output outputs/video/fox.mp4 --preset balanced
```

For image-to-video or video-to-video, add `--init-image PATH` or
`--init-video PATH`; `--strength` controls how much the initialized input is
changed. The generator supports `--segments` and `--overlap-frames` for
multi-segment extension. Use `scripts/generate_audio.py --help` or
`scripts/generate_video.py --help` to inspect all sampling controls.
