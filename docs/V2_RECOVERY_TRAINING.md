# V2 response-quality recovery

This stage repairs basic instruction-following behavior without overwriting the
completed v2 run. It uses a deterministic cleaner, a short shared base prompt
with the same safety/plain-text contract rendered during inference, and a balanced
six-domain mixture. Raw Wikipedia and rejected preference answers are intentionally
excluded from recovery SFT.

## 1. Prepare the cleaned dataset

```bash
.venv/bin/python scripts/prepare_recovery_sft.py \
  --output data/processed/recovery_sft
```

Review `data/processed/recovery_sft/summary.json` and manually sample every domain
before training. Generated JSONL files remain ignored by Git.

## 2. Start a new recovery stage

Use the best learned v2 weights, but create new optimizer and scheduler state:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/python scripts/train.py \
  --model-config configs/model.v2.gpu.yaml \
  --training-config configs/finetuning.v2.recovery.gpu.yaml \
  --tokenizer data/tokenizer-v2-extended \
  --init-from checkpoints/v2-finetuning/best.pt \
  --output checkpoints/v2-recovery/latest.pt \
  --best-output checkpoints/v2-recovery/best.pt
```

Do not use `--resume` for the first recovery run. Resume only an interrupted
recovery checkpoint:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/python scripts/train.py \
  --model-config configs/model.v2.gpu.yaml \
  --training-config configs/finetuning.v2.recovery.gpu.yaml \
  --tokenizer data/tokenizer-v2-extended \
  --resume checkpoints/v2-recovery/latest.pt \
  --output checkpoints/v2-recovery/latest.pt \
  --best-output checkpoints/v2-recovery/best.pt
```

## 3. Evaluate behavior

```bash
.venv/bin/python scripts/evaluate_benchmarks.py \
  --cases configs/evaluation.core.jsonl \
  --model-config configs/model.v2.gpu.yaml \
  --inference-config configs/inference.v2.yaml \
  --tokenizer data/tokenizer-v2-extended \
  --checkpoint checkpoints/v2-recovery/best.pt \
  --device cuda
```

Select checkpoints using both validation loss and fixed behavioral evaluations.
The generated manifests remain unreviewed until every upstream source license and
privacy status is manually approved.
