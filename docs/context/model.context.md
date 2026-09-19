# Compact Context: Model Architecture (`docs/context/model.context.md`)

> **AGENT CONTEXT PACK**: Load this file when implementing model architecture changes, attention modifications, layer normalization, or positional embeddings.

---

## 1. Authoritative Sources of Truth
- **Model Graph**: [`src/model/gpt.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/model/gpt.py) (`GPTModel`)
- **Layer Block**: [`src/model/transformer_block.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/model/transformer_block.py) (`TransformerBlock`)
- **Attention**: [`src/model/attention.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/model/attention.py) (`CausalSelfAttention`)
- **Feed-Forward**: [`src/model/feed_forward.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/model/feed_forward.py) (`FeedForward`)
- **Positions**: [`src/model/positional.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/model/positional.py) (`RotaryEmbedding`)
- **Loss**: [`src/model/loss.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/model/loss.py) (`ChunkedCrossEntropyLoss`)
- **Configuration**: [`src/model/config.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/model/config.py), [`configs/model.gpu.yaml`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/configs/model.gpu.yaml)

---

## 2. Active Model Shapes & Dimensions
- `vocab_size`: 40,000 (42,000 extended)
- `hidden_size`: 512
- `layers`: 16
- `heads`: 8 (query head dimension $d_k = 64$)
- `kv_heads`: 2 (GQA with 4 queries per key-value pair)
- `ffn_hidden_size`: 2048 (`swiglu`, `ffn_multiple_of: 128`)
- `norm_type`: `rms_norm` (Pre-norm, $\epsilon = 10^{-5}$)
- `position_type`: `rotary` (RoPE base 10000.0, scale 1.0)
- `tie_word_embeddings`: `true`
- `gradient_checkpointing`: `true`

---

## 3. Key Invariants
1. `hidden_size` must be divisible by `heads`.
2. `heads` must be divisible by `kv_heads`.
3. Rotary head dimension ($d_k = \text{hidden\_size} / \text{heads}$) must be even.
4. When vocabulary is extended, existing token indices (0 to 39,999) must never change.
5. In GPU training, `gradient_checkpointing` must remain enabled to prevent OOM on 4 GB VRAM.

---

## 4. Primary Verification Tests
```bash
.venv/bin/pytest tests/test_model_config.py tests/test_gpt.py tests/test_attention.py tests/test_transformer_block.py -q
```

