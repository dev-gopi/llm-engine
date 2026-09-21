# Compact Context: Training Subsystem (`docs/context/training.context.md`)

> **AGENT CONTEXT PACK**: Load this file when working on pretraining, fine-tuning, optimizers, learning rate schedules, checkpointing, or distributed sync.

---

## 1. Authoritative Sources of Truth
- **Trainer Engine**: [`src/training/trainer.py`](../../src/training/trainer.py) (`Trainer`)
- **Optimizer**: [`src/optim/adamw.py`](../../src/optim/adamw.py) (`AdamW`)
- **Schedulers**: [`src/optim/scheduler.py`](../../src/optim/scheduler.py) (`CosineAnnealingWithWarmup`)
- **EMA**: [`src/optim/ema.py`](../../src/optim/ema.py) (`EMAModel`)
- **Checkpointing**: [`src/training/checkpoint.py`](../../src/training/checkpoint.py)
- **Loss**: [`src/model/loss.py`](../../src/model/loss.py) (`ChunkedCrossEntropyLoss`)
- **Configs**: [`configs/pretraining.gpu.yaml`](../../configs/pretraining.gpu.yaml), [`configs/finetuning.gpu.yaml`](../../configs/finetuning.gpu.yaml)

---

## 2. Active Training Hyperparameters
- `batch_size`: 2
- `gradient_accumulation_steps`: 16 (effective batch size = 32)
- `max_sequence_length`: 512
- `learning_rate`: `5e-5`
- `weight_decay`: `0.1` (applied to 2D weight matrices only)
- `mixed_precision`: `fp16` or `bf16`
- `ema_decay`: `0.999`
- `z_loss_coefficient`: `0.0001`
- `chunk_size`: 128 tokens

---

## 3. Key Invariants
1. Never increase micro-batch size above 2 on a 4 GB GPU; use `gradient_accumulation_steps` to scale effective batch size.
2. Weight decay must exclude RMSNorm gains and linear biases.
3. Checkpoint saving is atomic (writes to temporary file, then renames to `.pt`).
4. Evaluator uses EMA weights if EMA is enabled; restores training weights immediately after evaluation.

---

## 4. Primary Verification Tests
```bash
.venv/bin/pytest tests/test_training_system.py tests/test_optim.py tests/test_loss.py -q
```

