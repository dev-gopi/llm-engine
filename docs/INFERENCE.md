# Inference Runtime & Memory Management (`docs/INFERENCE.md`)

*Authoritative Source: [`src/inference/generator.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/inference/generator.py), [`src/inference/paged_kv_cache.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/inference/paged_kv_cache.py), [`src/inference/sampler.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/inference/sampler.py), [`src/inference/quantization.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/inference/quantization.py)*

---

## 1. Generation Lifecycle & KV Caching

In naive generation, feeding the entire accumulated sequence $[t_1, \dots, t_k]$ to the Transformer at step $k$ leads to $O(N^2)$ recomputation. `llm-engine` implements two caching backends:

### 1.1 Tensor KV Cache (`src/model/kv_cache.py`)
Pre-allocates contiguous Key and Value tensors for single-request decoding. Fast and lightweight for single-stream generation.

### 1.2 Block-Paged KV Cache (`src/inference/paged_kv_cache.py`)
Inspired by operating system virtual memory paging:
- Allocates KV cache in fixed-size blocks (e.g. 16 or 32 tokens per block).
- Eliminates contiguous memory allocation requirements and memory fragmentation.
- Supports multi-sequence concurrent serving and prefix caching.

---

## 2. Sampling Strategies & Logit Warpers

The sampler in [`src/inference/sampler.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/inference/sampler.py) applies the following pipeline to raw next-token logits:

1. **Repetition Penalty**: Scales down logits of previously generated tokens:
   $$z_i \leftarrow \begin{cases} z_i / \theta & \text{if } z_i > 0 \\ z_i \cdot \theta & \text{if } z_i < 0 \end{cases}$$
2. **Temperature Scaling**: Adjusts entropy: $z_i \leftarrow z_i / T$.
3. **Top-K Truncation**: Retains only the $K$ highest-probability tokens.
4. **Top-P (Nucleus) Truncation**: Selects the smallest set of tokens whose cumulative probability exceeds $P$.
5. **Min-P Filtering**: Discards tokens with probability below $\text{min\_p} \times P_{\text{max}}$.
6. **Softmax & Categorical Sampling**: Draws token index from normalized distribution.

---

## 3. INT8 Quantization

[`src/inference/quantization.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/inference/quantization.py) provides dynamic INT8 quantization for linear projection layers:
- Scales weights: $W_{\text{int8}} = \text{round}(W_{\text{fp}} / s)$.
- Dequantizes dynamically during matrix multiplication: $Y = (X W_{\text{int8}}^T) \cdot s$.
- Yields a ~50% reduction in weight memory with negligible impact on generation perplexity.

---

## 4. Interactive Generation Commands

### Text Completion:
```bash
.venv/bin/python scripts/generate.py \
  --checkpoint checkpoints/pretraining/best.pt \
  --prompt "Once upon a time in a digital world," \
  --max-tokens 128 \
  --temperature 0.7
```

### Interactive Multi-Turn Chat:
```bash
.venv/bin/python scripts/chat.py \
  --checkpoint checkpoints/finetuning/best.pt
```

