# FSDP post-training wiring

This iteration enables the existing FSDP post-training checkpoint helpers in the production preference, reward-model, and offline-GRPO CLIs.

## Newly wired

- `scripts/train_dpo.py`: `distributed_strategy: fsdp` / `fsdp_hybrid` for DPO, IPO, ORPO and KTO.
- `scripts/train_reward_model.py`: FSDP / hybrid-FSDP reward-model training.
- `scripts/train_grpo.py`: FSDP / hybrid-FSDP offline GRPO policy training.
- Exact FSDP resume uses rank-sharded checkpoint directories with model, optimizer, scheduler, scaler and trainer state.
- New runs may still initialize from the existing single-file model checkpoints with `--init-from`.
- DDP and single-process checkpoint behavior remains unchanged.

## Launch pattern

Use `torchrun` and set `distributed_strategy: fsdp` (or `fsdp_hybrid`) in the selected training profile. FSDP requires at least two CUDA ranks in this engine.

Example:

```bash
torchrun --standalone --nproc-per-node=2 scripts/train_dpo.py \
  --model-config configs/model.gpu.yaml \
  --training-config configs/dpo.gpu.yaml \
  --tokenizer data/tokenizer-finetuning \
  --reference-checkpoint checkpoints/sft/best.pt \
  --init-from checkpoints/sft/best.pt
```

For an exact FSDP resume, pass the sharded output directory to `--resume`.

## Qualification boundary

The source wiring and non-CUDA regression tests are covered in CI. Real multi-GPU FSDP runs still require qualification on target CUDA hardware. The current shard format intentionally requires the same world size on resume; elastic resharding/consolidation remains a separate task.
