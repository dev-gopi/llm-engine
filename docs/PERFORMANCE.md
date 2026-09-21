# Performance & Memory Optimization (`docs/PERFORMANCE.md`)

*Authoritative Source: [`src/model/config.py`](../src/model/config.py), [`src/model/loss.py`](../src/model/loss.py), [`configs/model.gpu.yaml`](../configs/model.gpu.yaml)*

---

## 1. 4 GB VRAM Budget Allocation & Optimization

Training on a 4 GB GPU (NVIDIA RTX 3050) requires strict adherence to memory-reduction techniques:

| Component | Standard Transformer Cost | Optimized `llm-engine` Cost | Technique Used |
| :--- | :--- | :--- | :--- |
| **Model Weights** | ~325 MB (FP32) | **~165 MB** | Mixed Precision (FP16/BF16) + Tied Embeddings |
| **AdamW Optimizer**| ~1,300 MB | **~990 MB** | Decoupled decay; optimizer step on FP32 master weights |
| **Activations** | >3,500 MB (OOM!) | **~650 – 850 MB** | **Gradient Checkpointing** (activation recomputation) |
| **Loss Logits** | ~1,600 MB (OOM!) | **~120 MB** | **Chunked Cross-Entropy Loss** (chunk size 128) |
| **KV Cache** | ~64 MB / req (MHA) | **~16 MB / req** | **Grouped-Query Attention (GQA)** (2 KV heads) |

---

## 2. Gradient Checkpointing Mechanics

When `gradient_checkpointing: true` is enabled in `configs/model.gpu.yaml`:
- Intermediate activations within `TransformerBlock` layers are discarded during the forward pass.
- During the backward pass, each block recomputes its internal activations on-the-fly.
- **Trade-off**: Increases computation time by ~20–30%, but reduces activation memory by **over 75%**, allowing 16-layer training on 4 GB VRAM.

---

## 3. Chunked Loss Computation

Standard cross-entropy projects all $T=512$ sequence tokens to logits of size $V=40,000$ simultaneously:
$$\text{Tensor Size} = B \times T \times V \times 4\text{ bytes} = 2 \times 512 \times 40,000 \times 4 = 163.84\text{ MB (per step)}$$
Including gradient buffers and temporary softmax tensors, this peaks at >1.5 GB.

`ChunkedCrossEntropyLoss` splits the sequence dimension into chunks of $C=128$:
- Only one chunk of logits ($2 \times 128 \times 40,000$) is materialized at any instant.
- Peak logit allocation drops to **~40 MB**, completely preventing backward pass spikes.

---

## 4. KV Cache Memory Formula

For a sequence of length $L$, $L_y$ layers, $H_{kv}$ KV heads, and head dimension $d_k$:

$$\text{KV Cache Bytes (FP16)} = 2 \times L_y \times H_{kv} \times L \times d_k \times 2\text{ bytes}$$

For our architecture ($L_y=16$, $H_{kv}=2$, $d_k=64$):
$$\text{Bytes per token} = 2 \times 16 \times 2 \times 64 \times 2 = 8,192\text{ bytes} \approx 8\text{ KB/token}$$

- At context $L=512$: $\approx 4\text{ MB}$ per sequence.
- At context $L=1024$: $\approx 8\text{ MB}$ per sequence.
- Contrast with standard Multi-Head Attention ($H_{kv}=8$): would require $32\text{ MB}$ per sequence. GQA yields a **4x reduction** in cache footprint.

