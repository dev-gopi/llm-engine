# ADR-0009: Distributed Training Strategy: FSDP and Tensor Parallelism

## Status
`ACCEPTED`

## Context
As model configurations scale beyond 81M parameters to 1B, 7B, 30B, and 100B MoE, model weights, gradients, and optimizer states exceed single-GPU memory capacity. Standard Distributed Data Parallel (DDP) replicates full model weights on every device, rendering it incapable of training models larger than single-device VRAM.

## Decision
We adopted **Fully Sharded Data Parallelism (FSDP / ZeRO)** for multi-GPU training and **Tensor Parallelism (TP)** for intra-node projection sharding:
1. Use PyTorch FSDP to shard parameters, gradients, and optimizer states across data-parallel ranks.
2. Use Column-Parallel and Row-Parallel linear layers in [`src/inference/tensor_parallel.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/inference/tensor_parallel.py) for low-latency multi-GPU inference.

## Alternatives Considered
- **Pure Pipeline Parallelism (PP)**: Introduces pipeline bubbles and idle worker overhead.
- **DeepSpeed Dependency**: Adds heavy external runtime complexity; PyTorch native FSDP provides cleaner integration with zero C++ compilation steps.

## Consequences
- **Positive**: Enables training and serving models up to 30B+ across multi-GPU nodes with near-linear scaling efficiency.
- **Negative**: Inter-GPU communication overhead over NVLink or InfiniBand during all-gather and reduce-scatter phases.

## Related Files
- [`src/training/distributed.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/training/distributed.py)
- [`src/training/multinode.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/training/multinode.py)
- [`src/inference/tensor_parallel.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/inference/tensor_parallel.py)
- [`docs/SCALING.md`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/docs/SCALING.md)

