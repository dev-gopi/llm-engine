# Training Engine & Optimization (`docs/TRAINING.md`)

*Authoritative Source: [`src/training/trainer.py`](../src/training/trainer.py), [`src/optim/adamw.py`](../src/optim/adamw.py), [`src/optim/scheduler.py`](../src/optim/scheduler.py), [`src/model/loss.py`](../src/model/loss.py)*

---

## 1. Trainer Engine Architecture

The `Trainer` class in [`src/training/trainer.py`](../src/training/trainer.py) manages the training step:
1. **Micro-Batch Forward Pass**: Passes batches of size `batch_size: 2` through the model.
2. **Chunked Loss**: Uses `ChunkedCrossEntropyLoss` with sequence chunking (default chunk size 128) to compute cross-entropy without allocating full `[B, T, V]` logits in FP32.
3. **Loss Normalization**: Divides loss by `gradient_accumulation_steps` (16 or 32).
4. **Mixed Precision Backward**: Runs PyTorch Automatic Mixed Precision (`amp.autocast`) with `torch.cuda.amp.GradScaler`.
5. **Gradient Clipping**: Clips gradients at `gradient_clip_norm: 1.0` to avoid exploding gradients.
6. **Optimizer Step & Zero-Grad**: Updates weights every accumulation window and resets gradients with `set_to_none=True`.
7. **EMA Shadow Update**: Updates exponential moving average weights (`ema_decay: 0.999`).
8. **Periodic Validation**: Computes held-out domain losses and triggers early stopping or learning rate decay on loss plateaus.

---

## 2. Optimizer & Learning Rate Schedule

### AdamW Settings (`configs/pretraining.gpu.yaml`):
- `learning_rate`: `5e-5`
- `weight_decay`: `0.1` (applied only to 2D weight matrices; RMSNorm gains and biases excluded)
- `beta1`: `0.9`
- `beta2`: `0.95`
- `adam_epsilon`: `1e-8`

### Cosine Decay with Warmup:
$$\eta_t = \eta_{\text{min}} + \frac{1}{2} (\eta_{\text{max}} - \eta_{\text{min}}) \left(1 + \cos\left(\frac{t - T_{\text{warmup}}}{T_{\text{total}} - T_{\text{warmup}}} \pi\right)\right)$$

- 5% linear warmup ratio (`warmup_ratio: 0.05`).
- Minimum learning rate ratio (`min_lr_ratio: 0.1`).

---

## 3. Training Commands

### Pretraining:
```bash
.venv/bin/python scripts/train.py \
  --config configs/pretraining.gpu.yaml
```

### Instruction Fine-Tuning:
```bash
.venv/bin/python scripts/train.py \
  --config configs/finetuning.gpu.yaml
```

### DPO Preference Alignment:
```bash
.venv/bin/python scripts/train_dpo.py \
  --config configs/dpo.gpu.yaml
```

---

## 4. Checkpoint Resumption

To resume an interrupted training run:
```bash
.venv/bin/python scripts/train.py \
  --config configs/pretraining.gpu.yaml \
  --resume checkpoints/pretraining/latest.pt
```
The checkpoint loader restores:
- Model weights and tied embedding pointers.
- Optimizer momentum and variance states.
- Learning rate scheduler state and step index.
- Random number generator seeds (PyTorch, CUDA, Python, NumPy).

