# Direct model-growth and fine-tuning guide

> The filename is retained for existing links. Active configurations are
> unversioned; only `configs/tokenizer.v2.yaml` keeps a version suffix because
> tokenizer lineage is compatibility-sensitive.

This workflow upgrades an older 32K/10-layer checkpoint to the active
38K/16-layer GPU model. Keep the source checkpoint and tokenizer until the
grown model passes evaluation.

## 1. Build the append-only tokenizer

The tokenizer config reads `data/tokenizer-v2` and writes `data/tokenizer-v3`.

```bash
.venv/bin/python scripts/tokenize.py extend --config configs/tokenizer.v2.yaml
```

```bash
.venv/bin/python -c "from tokenizer.encoder import Tokenizer; t=Tokenizer.load('data/tokenizer-v3'); print({'vocab_size': t.vocab_size, 'base_vocab_size': t.base_vocab_size, 'append_only': bool(t.compatible_base_fingerprints)})"
```

The result must contain 38,000 tokens, match `configs/model.gpu.yaml`, and
report append-only compatibility.

## 2. Grow the source checkpoint

```bash
.venv/bin/python scripts/grow_checkpoint.py \
  --checkpoint checkpoints/source-finetuning/best.pt \
  --source-model-config configs/model.source.gpu.yaml \
  --target-model-config configs/model.gpu.yaml \
  --source-tokenizer data/tokenizer-v2 \
  --target-tokenizer data/tokenizer-v3 \
  --output checkpoints/grown/init.pt
```

This creates a separate checkpoint. Existing rows and layers are copied; new
vocabulary rows and layers are initialized for subsequent training.

## 3. Optional continued pretraining

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/python scripts/train.py \
  --model-config configs/model.gpu.yaml \
  --training-config configs/pretraining.gpu.yaml \
  --tokenizer data/tokenizer-v3 \
  --init-from checkpoints/grown/init.pt \
  --output checkpoints/pretraining/latest.pt \
  --best-output checkpoints/pretraining/best.pt
```

## 4. Start supervised fine-tuning

Use `--init-from` for the first invocation. It loads weights but creates fresh
optimizer, scheduler, sampler, and live-report state.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/python scripts/train.py \
  --model-config configs/model.gpu.yaml \
  --training-config configs/finetuning.gpu.yaml \
  --tokenizer data/tokenizer-v3 \
  --init-from checkpoints/pretraining/best.pt \
  --output checkpoints/finetuning/latest.pt \
  --best-output checkpoints/finetuning/best.pt
```

If continued pretraining was skipped, initialize from
`checkpoints/grown/init.pt` instead.

## 5. Resume an interrupted stage

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/python scripts/train.py \
  --model-config configs/model.gpu.yaml \
  --training-config configs/finetuning.gpu.yaml \
  --tokenizer data/tokenizer-v3 \
  --resume checkpoints/finetuning/latest.pt \
  --output checkpoints/finetuning/latest.pt \
  --best-output checkpoints/finetuning/best.pt
```

Do not combine `--resume` and `--init-from`. A new stage archives the previous
report; a resumed stage appends to its report history.

## 6. Monitor and evaluate

```bash
.venv/bin/python -m http.server 8000 --directory reports
```

Open `http://localhost:8000/training_report.html`. Choose the final checkpoint
using held-out validation and fixed behavioral evaluation, not training loss
alone.
