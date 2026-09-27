# Iteration 2: distributed preference training and IPO

## DDP DPO

Launch with torchrun and an ordinary DPO profile containing `distributed_strategy: ddp` (or omit it; DDP is the distributed default):

```bash
torchrun --standalone --nproc-per-node=2 scripts/train_dpo.py \
  --model-config configs/model.gpu.yaml \
  --training-config configs/dpo.gpu.yaml \
  --tokenizer data/tokenizer-finetuning \
  --reference-checkpoint checkpoints/sft/best.pt \
  --init-from checkpoints/sft/best.pt
```

Only rank 0 writes the checkpoint and prints the final JSON report. Each rank receives a deterministic distributed preference shard, while train/validation metrics are reduced across ranks.

## IPO

Use one of the IPO profiles or override the method on the CLI:

```bash
.venv/bin/python scripts/train_dpo.py \
  --method ipo \
  --model-config configs/model.gpu.yaml \
  --training-config configs/ipo.gpu.yaml \
  --tokenizer data/tokenizer-finetuning \
  --reference-checkpoint checkpoints/sft/best.pt \
  --init-from checkpoints/sft/best.pt
```

IPO uses the same preference-pair schema as DPO and records `training_type: ipo` in checkpoint metadata.

## SFT convenience entry point

```bash
.venv/bin/python scripts/train_sft.py \
  --model-config configs/model.gpu.yaml \
  --training-config configs/finetuning.gpu.yaml \
  --init-from checkpoints/pretraining/best.pt
```

The wrapper delegates to `scripts/train.py`; it does not fork a second SFT implementation.

## Remaining limitation

Preference training currently supports DDP, not FSDP. FSDP needs sharded checkpoint/reference-model handling and a memory-safe reference-policy strategy before it should be enabled.
