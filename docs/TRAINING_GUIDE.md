# Model training guide

> The filename is retained for existing links. The active model and training
> profiles now use unversioned filenames. `configs/tokenizer.v2.yaml` is the
> deliberate exception because tokenizer lineage affects checkpoint safety.

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

The active 38K model uses `data/tokenizer-v2`. The versioned config extends the
compatible 32K tokenizer at `data/tokenizer` without renumbering existing IDs:

Its source list matches all 29 training inputs in
`configs/finetuning.gpu.yaml`. Validation files remain excluded to prevent
held-out evaluation text from influencing tokenizer construction.

```bash
.venv/bin/python scripts/tokenize.py extend --config configs/tokenizer.v2.yaml
```

Verify the artifact:

```bash
.venv/bin/python scripts/tokenize.py inspect \
  --tokenizer data/tokenizer-v2 \
  "Hello, नमस्ते, বাংলা" --add-bos --add-eos
```

If no compatible `data/tokenizer` exists, train the 32K base with
`configs/tokenizer.yaml` first. Do not reuse checkpoints created with a
different tokenizer fingerprint.

## 4. Obtain the active model shape

The active GPU model is 38K/16-layer. Either grow a compatible older checkpoint
as described in [DIRECT_TRAINING_GUIDE.md](DIRECT_TRAINING_GUIDE.md), or
start it from scratch. Inspect it without allocating full weights:

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
  --tokenizer data/tokenizer-v2 \
  --output checkpoints/pretraining/latest.pt \
  --best-output checkpoints/pretraining/best.pt
```

To continue from model weights while starting a new optimizer stage, add
`--init-from PATH`. To resume an interrupted pretraining run, use:

```bash
.venv/bin/python scripts/train.py \
  --model-config configs/model.gpu.yaml \
  --training-config configs/pretraining.gpu.yaml \
  --tokenizer data/tokenizer-v2 \
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
  --tokenizer data/tokenizer-v2 \
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
  --tokenizer data/tokenizer-v2 \
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
  --tokenizer data/tokenizer-v2 \
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
  --tokenizer data/tokenizer-v2 \
  --checkpoint checkpoints/finetuning/best.pt \
  --device cuda

.venv/bin/python scripts/evaluate_benchmarks.py \
  --cases configs/evaluation.domains.jsonl \
  --model-config configs/model.gpu.yaml \
  --inference-config configs/inference.yaml \
  --tokenizer data/tokenizer-v2 \
  --checkpoint checkpoints/dpo/best.pt \
  --device cuda
```

## 10. Export

```bash
.venv/bin/python scripts/export.py \
  --model-config configs/model.gpu.yaml \
  --tokenizer data/tokenizer-v2 \
  --checkpoint checkpoints/dpo/best.pt \
  --format safetensors \
  --output exports/final/gopi.safetensors
```

Keep `model.yaml` and the copied tokenizer beside the exported weights.
