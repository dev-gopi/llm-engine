# Troubleshooting & Diagnostics Guide (`docs/TROUBLESHOOTING.md`)

This guide diagnoses common failures in `llm-engine` across training, CUDA memory, tokenization, checkpointing, and serving.

---

## 1. CUDA Out-of-Memory (OOM)

### Problem
Training or generation crashes with `torch.cuda.OutOfMemoryError: CUDA out of memory`.

### Symptoms
- Training halts during forward or backward pass.
- Peak allocated memory exceeds 4,096 MiB on RTX 3050.

### Cause
- `gradient_checkpointing` disabled in model config.
- `batch_size` too high (e.g. 4 or 8 instead of 2).
- Sequence length exceeds context without chunked loss.
- Stale background processes still occupying GPU VRAM.

### Diagnosis
Check active GPU memory allocation:
```bash
nvidia-smi
.venv/bin/python scripts/capabilities.py
```

### Solution
1. Verify `gradient_checkpointing: true` in `configs/model.gpu.yaml`.
2. Ensure `batch_size: 2` and compensate by increasing `gradient_accumulation_steps: 16` or `32`.
3. Verify `loss_reduction: mean` and chunked loss is active in `src/model/loss.py`.
4. Kill any orphan Python processes: `pkill -f train.py`.

### Prevention
Always run `scripts/capabilities.py` before starting long runs to verify available VRAM.

### Related Files
- [`configs/model.gpu.yaml`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/configs/model.gpu.yaml)
- [`src/model/loss.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/model/loss.py)

---

## 2. Non-Finite Gradients (NaN / Inf)

### Problem
Logs report `gradient_norm=nan` or `gradient_norm=inf`, and optimizer updates are discarded.

### Symptoms
- `nonfinite_updates` count increments in `reports/training_report.json`.
- PyTorch AMP GradScaler decreases its loss scale factor.

### Cause
- FP16 underflow/overflow on high learning rates or large logits.
- Missing `z_loss_coefficient` causing logits to drift to large magnitudes.

### Diagnosis
Inspect the training log for step-by-step gradient norms:
```bash
tail -n 50 logs/training.log
```

### Solution
1. If hardware supports it, switch to `mixed_precision: bf16` in training config (RTX 3050 supports BF16). BF16 has the dynamic range of FP32 and eliminates GradScaler scaling instability.
2. If using `fp16`, lower `learning_rate` from `5e-5` to `3e-5` and verify `gradient_clip_norm: 1.0`.
3. Confirm `z_loss_coefficient: 0.0001` is active in `configs/pretraining.gpu.yaml`.

### Prevention
Keep gradient clipping active and monitor z-loss regularization.

### Related Files
- [`src/optim/adamw.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/optim/adamw.py)
- [`src/training/trainer.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/training/trainer.py)

---

## 3. Tokenizer Fingerprint & Vocabulary Mismatch

### Problem
Resuming or fine-tuning fails with `ValueError: tokenizer fingerprint mismatch` or tensor shape mismatch on embedding weights.

### Symptoms
- Error loading state dict into `EmbeddingLayer` or `lm_head`.
- Model output contains scrambled characters or repetitive `<|unk|>`.

### Cause
- Checkpoint was trained with a different vocabulary size or different merge ranks.
- Retraining the tokenizer re-indexed base tokens.

### Diagnosis
Verify tokenizer compatibility with:
```bash
.venv/bin/pytest tests/test_vocabulary_compatibility.py -q
```

### Solution
- Ensure vocabulary expansion is **strictly append-only** (`data/tokenizer-finetuning/`).
- Never retrain a base tokenizer from scratch for an existing model checkpoint.

### Related Files
- [`src/model/vocabulary.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/model/vocabulary.py)
- [`src/tokenizer/bpe.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/tokenizer/bpe.py)

---

## 4. Repetitive or Incoherent Generation

### Problem
Autoregressive generation outputs looping text (e.g. "the the the...") or nonsensical token strings.

### Symptoms
- Model fails to produce `<|eos|>` stop token.
- High repetitive n-gram frequency.

### Cause
- Sampling temperature set to 0 without repetition penalty, or temperature too high (>1.2).
- Base pretraining checkpoint tested with conversational questions without SFT chat tuning.

### Diagnosis
Run test generation with repetition penalty:
```bash
.venv/bin/python scripts/generate.py \
  --checkpoint checkpoints/finetuning/best.pt \
  --temperature 0.7 \
  --repetition-penalty 1.15 \
  --top-p 0.9
```

### Solution
1. Apply `repetition_penalty: 1.15` in `configs/inference.yaml`.
2. For conversational prompts, ensure the fine-tuned chat checkpoint (`checkpoints/finetuning/best.pt`) is loaded, not the raw pretraining checkpoint.

### Related Files
- [`src/inference/sampler.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/inference/sampler.py)
- [`src/inference/generator.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/inference/generator.py)

---

## 5. Build & Packaging Metadata Issues (`*.egg-info`)

### Problem
Untracked `src/llm_engine.egg-info` appears after running editable pip install, cluttering the working tree.

### Symptoms
- Build metadata or stale dependency records persist in `src/`.

### Cause
- `pyproject.toml` configures setuptools package search with `where = ["src"]`. When `pip install -e .` runs, setuptools generates `<pkg>.egg-info` in `src/`.

### Solution
- `*.egg-info/` is already ignored in `.gitignore`.
- Remove safely at any time: `rm -rf src/llm_engine.egg-info`.
- Tests run directly using `.venv/bin/pytest` via `PYTHONPATH=src:.` without requiring egg-info metadata.

### Related Files
- [`pyproject.toml`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/pyproject.toml)
- [`.gitignore`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/.gitignore)
