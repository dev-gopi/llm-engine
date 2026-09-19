# ADR-0005: Training Strategy for 4 GB Consumer GPUs

## Status
`ACCEPTED`

## Context
The development environment operates on an NVIDIA GeForce RTX 3050 (4,096 MiB VRAM). Training an 81.3M parameter Transformer with standard settings (batch size 16, full activation retention, monolithic cross-entropy) instantly triggers CUDA Out-of-Memory (OOM) errors.

## Decision
We established a strict three-pillar memory architecture for all active GPU training configurations:
1. **Mandatory Gradient Checkpointing**: Recompute intermediate activations during backward pass.
2. **Chunked Cross-Entropy Loss**: Compute loss in sequence chunks of $C=128$, preventing full $[B, T, V]$ logit materialization.
3. **Small Micro-Batches with Large Gradient Accumulation**: Set `batch_size: 2` and `gradient_accumulation_steps: 16` or `32`.

## Alternatives Considered
- **Reducing Model Size (<20M params)**: Reduces representation capacity and ruins conversational coherence.
- **CPU Offloading during training**: Severely throttles training throughput due to PCIe bandwidth bottlenecks.

## Consequences
- **Positive**: Stable, non-crashing training of an 81.3M parameter model on 4 GB VRAM with ~1.3 GB remaining margin.
- **Negative**: Gradient checkpointing incurs ~20% higher wall-clock computation time per epoch.

## Related Files
- [`configs/model.gpu.yaml`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/configs/model.gpu.yaml)
- [`configs/pretraining.gpu.yaml`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/configs/pretraining.gpu.yaml)
- [`src/model/loss.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/model/loss.py)

