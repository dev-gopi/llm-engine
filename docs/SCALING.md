# Scaling Architecture: From 80M to 1T Parameters (`docs/SCALING.md`)

*Authoritative Source: [`src/training/model_growth.py`](../src/training/model_growth.py), [`src/training/peft.py`](../src/training/peft.py), [`src/training/multinode.py`](../src/training/multinode.py), [`src/training/elastic.py`](../src/training/elastic.py), [`src/training/distributed_checkpoint.py`](../src/training/distributed_checkpoint.py), [`src/inference/tensor_parallel.py`](../src/inference/tensor_parallel.py)*

This document defines the configuration-driven scaling architecture of `llm-engine`, detailing the transition from single-GPU consumer hardware (81.3M parameters on 4 GB VRAM) up to multi-node cluster training (1B, 7B, 30B, 100B MoE, and 1T dense models).

---

## 1. Hardware & Model Capacity Tiers

The framework scales across 5 distinct hardware and model tiers:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ TIER 5: Ultra / Supercluster (100B MoE – 1T Dense)                                     │
│ 128 – 1,024 x H100/B200 GPUs | 3D Parallelism (TP + PP + FSDP) + Expert Parallelism    │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ TIER 4: Large Enterprise (30B – 70B Dense)                                             │
│ 4 – 8 x H100 (80GB) | FSDP ZeRO-3 + Tensor Parallelism + FlashAttention                │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ TIER 3: Medium Workstation (7B – 14B Dense)                                            │
│ 1 – 2 x A100/H100 (80GB) or Dual 24GB | FSDP Full Shard + QLoRA / 4-bit PEFT           │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ TIER 2: Prosumer / Desktop (1B – 3B Dense)                                             │
│ 1 x RTX 3090 / 4090 (24GB) or Apple Silicon 32GB+ | Single-Device AdamW / LoRA          │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ TIER 1: Consumer / Edge Baseline (81.3M – 125M) [ACTIVE CURRENT]                       │
│ 1 x NVIDIA RTX 3050 (4 GB VRAM) | DDP, Gradient Checkpointing, Chunked Cross-Entropy   │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Parameter Sizing & Memory Formulas

Parameter count and memory calculations are performed with zero weight allocations via [`src/model/config.py:estimate_model_size`](../src/model/config.py):

| Model Spec | Config File | Layers | Hidden | Heads / KV | FFN Dim | Dense Params | Active Params | FP16 Weights | AdamW State | Target Hardware |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **81.3M (Base)** | `configs/model.gpu.yaml` | 16 | 512 | 8 / 2 | 2,048 | 81.3M | 81.3M | ~165 MB | ~990 MB | 1x RTX 3050 (4 GB) |
| **82.3M (Chat)** | `data/tokenizer-finetuning/`| 16 | 512 | 8 / 2 | 2,048 | 82.3M | 82.3M | ~167 MB | ~1,005 MB | 1x RTX 3050 (4 GB) |
| **1B (Future)** | `configs/text/model.future.1b.yaml`| 24 | 2,048 | 16 / 4 | 8,192 | ~1.1B | ~1.1B | ~2.2 GB | ~13.2 GB | 1x RTX 4090 (24 GB) |
| **7B (Future)** | `configs/text/model.future.7b.yaml`| 32 | 4,096 | 32 / 8 | 14,336| ~6.8B | ~6.8B | ~13.6 GB | ~81.6 GB | 2x A100 (80 GB) or 4x 4090 |
| **30B (Future)** | `configs/text/model.future.30b.yaml`| 60 | 6,656 | 52 / 8 | 24,064| ~31.2B | ~31.2B | ~62.4 GB | ~374 GB | 8x A100 / H100 (80 GB) |
| **100B MoE** | `configs/scaling/model.moe-100b.yaml`| 48 | 4,096 | 32 / 8 | 11,264| ~98.4B | ~14.2B | ~196 GB | ~1.18 TB | 8–16x H100 (80 GB) |
| **1T Dense** | `configs/scaling/model.1t.yaml` | 164 | 24,576 | 192 / 8 | 65,536| ~999.8B | ~999.8B | ~2.0 TB | ~12.0 TB | 512–1,024x H100 Clusters |

---

## 3. Distributed Parallelism Dimensions

To scale beyond single-device memory limits, `llm-engine` incorporates five orthogonal parallelism axes:

### 3.1 Fully Sharded Data Parallelism (FSDP / ZeRO)
Implemented via PyTorch FSDP in [`src/training/distributed.py`](../src/training/distributed.py):
- **ZeRO-1**: Shards optimizer states across data-parallel ranks (reduces memory by 4x).
- **ZeRO-2**: Shards optimizer states and gradients across ranks.
- **ZeRO-3 (Full Shard)**: Shards optimizer states, gradients, and model parameters. Tensors are all-gathered only on-demand during the forward/backward pass and immediately freed.

### 3.2 Tensor Parallelism (TP)
Implemented in [`src/inference/tensor_parallel.py`](../src/inference/tensor_parallel.py):
- Splits weight matrices within an individual Transformer layer across $K$ GPUs:
  - Attention Query/Key/Value: Column-parallel linear projection (splits heads across devices).
  - Attention Output: Row-parallel linear projection with all-reduce sum across devices.
  - SwiGLU FFN: Column-parallel gate/up projections; Row-parallel down projection.
- Eliminates memory bottlenecks for layers that exceed a single device's VRAM.

### 3.3 Pipeline Parallelism (PP)
- Partitions the 16 to 164 `TransformerBlock` layers across distinct physical devices.
- Uses micro-batching (1F1B schedule) to minimize the pipeline bubble.

### 3.4 Expert Parallelism (EP) for MoE
- In Mixture-of-Experts layers (`ffn_type: moe` in [`src/model/feed_forward.py`](../src/model/feed_forward.py)), routing is computed globally, while individual experts reside on distinct GPUs.
- Allows scaling total parameters to 100B+ while keeping active compute per token at ~14B.

---

## 4. Progressive Model Growth (Warm Starts Without Cold Training)

Instead of training 1B or 7B models from random Gaussian noise (which wastes millions of compute hours), [`src/training/model_growth.py`](../src/training/model_growth.py) enables **Progressive Model Growth**:

```text
81.3M Base Model (16 layers, width 512)
          │
          ▼ grow_model() via Net2Net identity expansion
162.6M Intermediate Model (32 layers, width 512)
          │ newly appended layers initialized as identity residuals:
          │ W_out = 0, W_ffn_down = 0 -> exact function preservation
          ▼ Short continued pretraining
Grown Stable Model
```

- **Mathematical Invariant**: Newly added blocks have their output projection weights initialized to exact zeros. The expanded model's initial outputs are 100% mathematically identical to the smaller parent model's outputs.

---

## 5. Parameter-Efficient Fine-Tuning (PEFT / LoRA)

For fine-tuning 7B to 70B models without updating billions of base parameters, [`src/training/peft.py`](../src/training/peft.py) provides native **Low-Rank Adaptation (LoRA)**:

$$W = W_{\text{base}} + \frac{\alpha}{r} (A \cdot B)$$

- $W_{\text{base}} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ is frozen (`requires_grad = False`).
- $A \in \mathbb{R}^{r \times d_{\text{in}}}$ is initialized with Kaiming uniform.
- $B \in \mathbb{R}^{d_{\text{out}} \times r}$ is initialized with exact zeros.
- Trainable parameters represent **<0.5%** of base weights, enabling fine-tuning of 7B-class models on a single consumer GPU.

### Applying LoRA:
```python
from training.peft import apply_lora

lora_summary = apply_lora(model, {
    "lora_rank": 8,
    "lora_alpha": 16.0,
    "lora_dropout": 0.05,
    "lora_target_modules": ["q_proj", "v_proj"]
})
print(f"Trainable params: {lora_summary['trainable_parameters']:,}")
```

---

## 6. Cluster Fault Tolerance & Elastic Scaling

Large-scale distributed pretraining across hundreds of GPUs requires automated failure recovery:
- **Elastic Rendezvous (`src/training/elastic.py`)**: Heartbeat monitor that detects node death and re-rendezvous surviving ranks automatically.
- **Preflight Collective Validation (`src/training/multinode.py`)**: Runs collective all-reduce barriers at job startup to detect silent network stalls before training starts.
- **Distributed Sharded Checkpointing (`src/training/distributed_checkpoint.py`)**: Each rank serializes only its local slice of the model in parallel, reducing multi-terabyte checkpoint write times from minutes to seconds.
