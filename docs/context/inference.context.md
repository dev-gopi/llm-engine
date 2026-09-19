# Compact Context: Inference Subsystem (`docs/context/inference.context.md`)

> **AGENT CONTEXT PACK**: Load this file when working on generation, sampling, KV caching, quantization, or speculative decoding.

---

## 1. Authoritative Sources of Truth
- **Generator**: [`src/inference/generator.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/inference/generator.py) (`AutoregressiveGenerator`)
- **Sampler**: [`src/inference/sampler.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/inference/sampler.py) (`Sampler`)
- **Paged KV Cache**: [`src/inference/paged_kv_cache.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/inference/paged_kv_cache.py) (`PagedKVCache`)
- **Quantization**: [`src/inference/quantization.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/inference/quantization.py) (`quantize_dynamic_int8`)
- **Chat Session**: [`src/inference/chat_session.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/inference/chat_session.py) (`ChatSession`)
- **Runtime Config**: [`configs/inference.yaml`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/configs/inference.yaml)

---

## 2. Key Operational Mechanisms
- **Prefill vs. Decode**: Prefill processes prompt tokens in parallel; decode generates autoregressively 1 token per step appending to the KV cache.
- **Sampling Pipeline**: `Logits` $\rightarrow$ Repetition Penalty $\rightarrow$ Temperature $\rightarrow$ Top-K $\rightarrow$ Top-P $\rightarrow$ Softmax $\rightarrow$ Categorical Sample.
- **Quantization**: Linear projection layers can be dynamically quantized to INT8 with scaling factors, halving memory with <1% perplexity difference.

---

## 3. Key Invariants
1. KV cache blocks are immutable once finalized for prefix caching.
2. Generating past `max_position` triggers graceful stop or context window compaction.
3. Quantization must never quantize RMSNorm or Embedding layers to avoid precision collapse.

---

## 4. Primary Verification Tests
```bash
.venv/bin/pytest tests/test_generation.py tests/test_inference_precision.py tests/test_chat_session.py -q
```

