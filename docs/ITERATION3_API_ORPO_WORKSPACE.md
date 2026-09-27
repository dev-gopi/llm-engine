# Iteration 3 — ORPO and API/runtime parity hardening

This iteration builds directly on the distributed-preference + semantic-cache tree.

Implemented:

- ORPO preference training with paired chosen/rejected data and no reference model requirement;
- CPU/GPU ORPO profiles and `--method orpo` support in `scripts/train_dpo.py`;
- OpenAI-style presence/frequency penalties across native single, streaming, and batched decoding;
- forwarding of presence/frequency penalties through the OpenAI-compatible external backend;
- embeddings `encoding_format=base64` using little-endian float32 wire format;
- reduced embedding output dimensions from 1 through the backend native dimension;
- reviewed workspace patch creation/deletion while retaining path-boundary checks and `git apply --check` validation.

Preference training remains DDP-only in this CLI. FSDP preference checkpointing is intentionally left on the missing-feature list until sharded policy/reference and optimizer state handling can be verified on CUDA hardware.

## ORPO example

```bash
.venv/bin/python scripts/train_dpo.py \
  --method orpo \
  --model-config configs/model.gpu.yaml \
  --training-config configs/orpo.gpu.yaml \
  --tokenizer data/tokenizer-finetuning \
  --init-from checkpoints/sft/best.pt
```

Distributed ORPO uses the same DDP launch path:

```bash
torchrun --standalone --nproc-per-node=2 scripts/train_dpo.py \
  --method orpo \
  --model-config configs/model.gpu.yaml \
  --training-config configs/orpo.gpu.yaml \
  --tokenizer data/tokenizer-finetuning \
  --init-from checkpoints/sft/best.pt
```
