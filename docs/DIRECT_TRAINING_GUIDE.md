# Direct training and fine-tuning guide

The base 40K tokenizer is saved at `data/tokenizer`. The configured FineWeb-Edu
and CodeParrot sources create a safe append-only extension at
`data/tokenizer-finetuning`, preserving every base token ID.

## 1. Train the tokenizer

For a new model family only, train the base tokenizer from scratch:

```bash
.venv/bin/python scripts/tokenize.py train --config configs/tokenizer.yaml
```

For an existing 40K checkpoint, do not retrain the tokenizer. Build every
configured extension:

```bash
.venv/bin/python scripts/tokenize.py extend --config configs/tokenizer.yaml
```

Build only one named extension when needed:

```bash
.venv/bin/python scripts/tokenize.py extend \
  --config configs/tokenizer.yaml --extension finetuning
```

Extension names are user-defined labels. Each item declares its own
`base_tokenizer`, so extensions can be independent or chained by using an
earlier item's `output_dir` as the next item's `base_tokenizer`.

## 2. Start pretraining

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/python scripts/train.py \
  --model-config configs/model.gpu.yaml \
  --training-config configs/pretraining.gpu.yaml \
  --tokenizer data/tokenizer \
  --output checkpoints/pretraining/latest.pt \
  --best-output checkpoints/pretraining/best.pt
```

## 3. Start supervised fine-tuning

Use `--init-from` for the first invocation. It loads weights but creates fresh
optimizer, scheduler, sampler, and live-report state.

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

## 4. Resume an interrupted stage

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/python scripts/train.py \
  --model-config configs/model.gpu.yaml \
  --training-config configs/finetuning.gpu.yaml \
  --tokenizer data/tokenizer-finetuning \
  --resume checkpoints/finetuning/latest.pt \
  --output checkpoints/finetuning/latest.pt \
  --best-output checkpoints/finetuning/best.pt
```

Do not combine `--resume` and `--init-from`. A new stage archives the previous
report; a resumed stage appends to its report history.

## 5. Monitor and evaluate

```bash
.venv/bin/python -m http.server 8000 --directory reports
```

Open `http://localhost:8000/training_report.html`. Choose the final checkpoint
using held-out validation and fixed behavioral evaluation, not training loss
alone.
