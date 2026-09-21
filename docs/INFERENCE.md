# Inference Runtime & Memory Management (`docs/INFERENCE.md`)

*Authoritative Source: [`src/inference/generator.py`](../src/inference/generator.py), [`src/inference/paged_kv_cache.py`](../src/inference/paged_kv_cache.py), [`src/inference/sampler.py`](../src/inference/sampler.py), [`src/inference/quantization.py`](../src/inference/quantization.py)*

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

The sampler in [`src/inference/sampler.py`](../src/inference/sampler.py) applies the following pipeline to raw next-token logits:

1. **Repetition Penalty**: Scales down logits of previously generated tokens:
   $$z_i \leftarrow \begin{cases} z_i / \theta & \text{if } z_i > 0 \\ z_i \cdot \theta & \text{if } z_i < 0 \end{cases}$$
2. **Temperature Scaling**: Adjusts entropy: $z_i \leftarrow z_i / T$.
3. **Top-K Truncation**: Retains only the $K$ highest-probability tokens.
4. **Top-P (Nucleus) Truncation**: Selects the smallest set of tokens whose cumulative probability exceeds $P$.
5. **Min-P Filtering**: Discards tokens with probability below $\text{min\_p} \times P_{\text{max}}$.
6. **Softmax & Categorical Sampling**: Draws token index from normalized distribution.

---

## 3. INT8 Quantization

[`src/inference/quantization.py`](../src/inference/quantization.py) provides dynamic INT8 quantization for linear projection layers:
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


---

## 5. Q1_0 Research Packing and GGUF Delegation

`src/inference/quantization.py` now includes a Q1_0-style binary research
representation: one sign bit per weight with one FP16 scale for each group of
128 weights, or 1.125 effective bits/weight. `scripts/export.py --weight-dtype
q1_0` writes a self-describing safetensors artifact. This format is intended for
engine experiments and is **not** represented as byte-for-byte GGUF
compatibility.

For actual GGUF models and runtime-specific low-bit kernels, use the delegated
backend. A local llama.cpp server can be started with:

```bash
python scripts/serve_gguf.py /path/to/Bonsai-27B-Q1_0.gguf \
  --context 8192 --gpu-layers 99 --cache-type-k q8_0 --cache-type-v q8_0
```

Then run the Gopi server with the external backend selected:

```bash
GOPI_BACKEND=llama_cpp \
GOPI_EXTERNAL_BASE_URL=http://127.0.0.1:8080 \
GOPI_EXTERNAL_MODEL=local-model \
GOPI_EXTERNAL_CONTEXT_LENGTH=8192 \
python scripts/serve.py
```

The adapter forwards OpenAI-compatible chat generation and streaming while the
Gopi API continues to provide its authentication, client compatibility,
observability and playground surface.

## 6. Deployment Resource Planning

Use the analytical planner before loading a large checkpoint:

```bash
python scripts/plan_deployment.py \
  --model-config configs/model.hybrid.gpu.yaml \
  --context-length 8192 --memory-gib 4
```

The same information is available at
`GET /v1/models/{model_id}/resources`. Estimates include resident weights,
context-growing full-attention KV state, fixed-size linear-attention recurrent
state, batch size, precision and a configurable safety margin. Backend
workspaces and fragmentation are intentionally described as unmeasured rather
than hidden inside a false precision claim.
