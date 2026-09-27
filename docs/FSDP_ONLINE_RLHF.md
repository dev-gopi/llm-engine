# FSDP online RLHF

The online PPO and online GRPO entry points now support `distributed_strategy: fsdp` and `fsdp_hybrid` under `torchrun`.

Example PPO:

```bash
torchrun --standalone --nproc-per-node=2 scripts/train_ppo.py \
  --model-config configs/model.gpu.yaml \
  --training-config configs/ppo.fsdp.yaml \
  --reference-checkpoint checkpoints/sft/best.pt \
  --reward-checkpoint checkpoints/reward-model/best.pt \
  --init-from checkpoints/sft/best.pt
```

Example online GRPO:

```bash
torchrun --standalone --nproc-per-node=2 scripts/train_grpo_online.py \
  --model-config configs/model.gpu.yaml \
  --training-config configs/grpo.online.fsdp.yaml \
  --reference-checkpoint checkpoints/sft/best.pt \
  --reward-checkpoint checkpoints/reward-model/best.pt \
  --init-from checkpoints/sft/best.pt
```

FSDP online-RLHF checkpoints are rank sharded and exact resume currently requires the same world size. Rollout generation uses ordinary replica models synchronized collectively from the FSDP training weights, avoiding token-by-token generation directly through the sharded wrapper.
