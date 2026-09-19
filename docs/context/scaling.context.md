# Compact Context: Scaling & Distributed Infrastructure (`docs/context/scaling.context.md`)

> **AGENT CONTEXT PACK**: Load this file when implementing multi-GPU training, FSDP, model growth, PEFT/LoRA, tensor parallelism, or scaling models from 80M to 1B, 7B, 30B, 100B MoE, or 1T.

---

## 1. Authoritative Sources of Truth
- **Model Growth**: [`src/training/model_growth.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/training/model_growth.py) (`grow_model`)
- **PEFT / LoRA**: [`src/training/peft.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/training/peft.py) (`LoRALinear`, `apply_lora`)
- **Multi-Node Topology**: [`src/training/multinode.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/training/multinode.py) (`topology_from_environment`, `validate_collectives`)
- **Elastic Training**: [`src/training/elastic.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/training/elastic.py)
- **Sharded Checkpoints**: [`src/training/distributed_checkpoint.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/training/distributed_checkpoint.py)
- **Tensor Parallel Inference**: [`src/inference/tensor_parallel.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/inference/tensor_parallel.py)
- **Scaling Configurations**: [`configs/scaling/`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/configs/scaling/), [`configs/text/`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/configs/text/)
- **Full Architecture Guide**: [`docs/SCALING.md`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/docs/SCALING.md)

---

## 2. Scaling Hierarchy (5 Tiers)
1. **Tier 1 (81M)**: 1x RTX 3050 (4 GB) — Active baseline; DDP + chunked loss + gradient checkpointing.
2. **Tier 2 (1B - 3B)**: 1x RTX 3090/4090 (24 GB) — FSDP / single-device AdamW + LoRA.
3. **Tier 3 (7B - 14B)**: 1-2x A100/H100 (80 GB) — FSDP ZeRO-3 Full Shard + QLoRA.
4. **Tier 4 (30B - 70B)**: 4-8x H100 (80 GB) — FSDP + Tensor Parallelism.
5. **Tier 5 (100B MoE - 1T)**: 128-1024x GPU Clusters — 3D Parallelism (TP + PP + FSDP) + Expert Parallelism.

---

## 3. Key Invariants
1. **Model Growth Identity Invariant**: When expanding layer count using `grow_model()`, new blocks MUST have their output projections zero-initialized so the initial grown model function equals the parent model function.
2. **LoRA Freezing**: `apply_lora()` must freeze all base model parameters (`requires_grad = False`) and initialize adapter $B$ to zero.
3. **Planning Profiles**: Models marked `planning_only: true` in `configs/scaling/` (e.g. `model.1t.yaml`) cannot allocate weights directly without distributed sharding.

---

## 4. Primary Verification Tests
```bash
.venv/bin/pytest tests/test_scaling_features.py tests/test_model_growth.py tests/test_peft.py tests/test_tensor_parallel.py tests/test_multinode.py -q
```

