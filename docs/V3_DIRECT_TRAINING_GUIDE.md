# V3 direct fine-tuning guide

This workflow grows the trained 10-layer v2 model into the 16-layer, 38K
vocabulary v3 model and starts supervised fine-tuning without an intervening
continued-pretraining stage. Keep all v2 artifacts for comparison and rollback.
The complete source mixture and its review status are documented in the
[dataset catalog](DATASET_CATALOG.md).

Run every command from the repository root.

## 1. Create the v3 tokenizer

The v3 tokenizer is a verified append-only extension of
`data/tokenizer-v2-extended`. Existing v2 token IDs remain unchanged.

```bash
.venv/bin/python scripts/tokenize.py extend \
  --config configs/tokenizer.v3.extension.yaml
```

The command must report:

```text
old_vocab_size: 34000
new_vocab_size: 38000
```

Verify the saved tokenizer and its lineage:

```bash
.venv/bin/python -c "from tokenizer.encoder import Tokenizer; t=Tokenizer.load('data/tokenizer-v3-extended-38k'); print({'vocab_size': t.vocab_size, 'base_vocab_size': t.base_vocab_size, 'append_only': bool(t.compatible_base_fingerprints)})"
```

Expected values are `vocab_size: 38000`, `base_vocab_size: 32000`, and
`append_only: True`. Do not grow the model if the vocabulary is smaller than
38,000; its size must match `configs/model.v3.gpu.yaml`.

## 2. Grow the v2 checkpoint

Select the best v2 fine-tuning checkpoint, not merely the latest checkpoint.
The converter copies the ten trained layers and the existing vocabulary rows,
then adds six identity-like layers and mean-initialized vocabulary rows.

```bash
.venv/bin/python scripts/grow_checkpoint.py \
  --checkpoint checkpoints/v2-finetuning/best.pt \
  --source-model-config configs/model.v2.55m-source.yaml \
  --target-model-config configs/model.v3.gpu.yaml \
  --source-tokenizer data/tokenizer-v2-extended \
  --target-tokenizer data/tokenizer-v3-extended-38k \
  --output checkpoints/v3-direct/init.pt
```

This operation creates a separate checkpoint and does not overwrite v2.

## 3. Start direct v3 fine-tuning

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/python scripts/train.py \
  --model-config configs/model.v3.gpu.yaml \
  --training-config configs/finetuning.v3.gpu.yaml \
  --tokenizer data/tokenizer-v3-extended-38k \
  --init-from checkpoints/v3-direct/init.pt \
  --output checkpoints/v3-direct-finetuning/latest.pt \
  --best-output checkpoints/v3-direct-finetuning/best.pt
```

Use `--init-from` only for this first v3 run. It starts a fresh optimizer and
scheduler while loading the grown model weights.

## 4. Resume an interrupted v3 run

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/python scripts/train.py \
  --model-config configs/model.v3.gpu.yaml \
  --training-config configs/finetuning.v3.gpu.yaml \
  --tokenizer data/tokenizer-v3-extended-38k \
  --resume checkpoints/v3-direct-finetuning/latest.pt \
  --output checkpoints/v3-direct-finetuning/latest.pt \
  --best-output checkpoints/v3-direct-finetuning/best.pt
```

Do not combine `--resume` with `--init-from`. Resume restores the complete v3
training state, so the model, tokenizer, and training configuration must match
the interrupted run.

## 5. Optional alternative training routes

### Keep the 10-layer v2 architecture

To use the expanded datasets and 38K tokenizer without adding six layers,
start a separate optimizer stage directly from the v2 best checkpoint:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/python scripts/train.py \
  --model-config configs/model.v2.gpu.yaml \
  --training-config configs/finetuning.v2.expanded.gpu.yaml \
  --tokenizer data/tokenizer-v3-extended-38k \
  --init-from checkpoints/v2-finetuning/best.pt \
  --output checkpoints/v2-expanded-finetuning/latest.pt \
  --best-output checkpoints/v2-expanded-finetuning/best.pt
```

This is safer than adding layers because the complete transformer stack is
already trained. It still initializes the newly appended vocabulary rows.

### Continued pretraining before v3 SFT

For the more reliable v3 growth route, write the grown checkpoint to
`checkpoints/v3-grown/init.pt`, then train the new layers on the causal-language
modeling objective:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/python scripts/train.py \
  --model-config configs/model.v3.gpu.yaml \
  --training-config configs/pretraining.v3.grown.gpu.yaml \
  --tokenizer data/tokenizer-v3-extended-38k \
  --init-from checkpoints/v3-grown/init.pt \
  --output checkpoints/v3-pretraining/latest.pt \
  --best-output checkpoints/v3-pretraining/best.pt
```

After pretraining stabilizes, run v3 SFT with
`checkpoints/v3-pretraining/best.pt` as `--init-from` and use separate v3
fine-tuning output paths.

## 6. Monitor and evaluate

Training automatically updates the live report. Serve it from the report
directory:

```bash
.venv/bin/python -m http.server 8000 --directory reports
```

Open `http://localhost:8000/training_report.html`. Retain the v2 model until v3
shows better held-out validation and generation quality; direct SFT is more
experimental than continued pretraining because the six new layers initially
see only instruction data.
