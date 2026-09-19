# Target-State Architecture & Capabilities (`docs/TARGET_STATE.md`)

This document defines the intended upgraded production platform. It clearly delineates **CURRENT** state, **TARGET** state, and the concrete **MIGRATION** path for each capability area.

---

## 1. Model Architecture & Attention Mechanism

### CURRENT
- Dense causal Transformer decoder with 16 identical full-attention layers.
- Standard Grouped-Query Attention (GQA, 8 query heads, 2 KV heads).
- $O(N^2)$ attention computation limits sequence length to 1024 tokens.

### TARGET
- **Hybrid Attention Transformer**:
  - Mixture of linear/recurrent attention layers (e.g., 75% linear attention) and full attention layers (25% full attention).
  - Sliding-window local attention for intermediate tokens; global attention for system prompt tokens (attention sinks).
  - Multi-Token Prediction (MTP) auxiliary head for speculative decoding acceleration.
- **Sparse MoE (Mixture of Experts)**:
  - Configurable sparse feed-forward layers with Top-K routing and load-balancing auxiliary loss.

### MIGRATION
1. Implement linear attention sublayer in [`src/model/attention.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/model/attention.py) alongside GQA.
2. Add layer-wise attention type configuration in [`src/model/transformer_block.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/model/transformer_block.py).
3. Validate gradient flow and backward pass with tests in `tests/test_attention.py`.
4. Train MTP prediction heads using auxiliary loss without modifying the base backbone weights.

---

## 2. Context Length & Positional Encoding

### CURRENT
- `max_position: 1024` with standard Rotary Position Embeddings (RoPE, `rope_base: 10000.0`, `rope_scale: 1.0`).
- Pretraining and fine-tuning datasets formatted to 512 tokens.

### TARGET
- Stepped context length expansion: **1K $\rightarrow$ 2K $\rightarrow$ 4K $\rightarrow$ 8K $\rightarrow$ 16K**.
- RoPE scaling strategies:
  - NTK-Aware RoPE scaling.
  - YaRN (Yet another RoPE extensioN) interpolation for preserving high-frequency resolution.
- Passkey retrieval and "Needle in a Haystack" benchmark validation.

### MIGRATION
1. Add `rope_scaling_type` (`linear`, `ntk`, `yarn`) to [`src/model/positional.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/model/positional.py).
2. Create synthetic long-context evaluation suite in [`src/evaluation/benchmarks.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/evaluation/benchmarks.py).
3. Incrementally fine-tune on packed multi-turn dialogues with extended sequence lengths.

---

## 3. Tokenizer & Chat Formatting

### CURRENT
- 40,000 base BPE vocabulary; extended to 42,000 for fine-tuning.
- Special tokens: `<|pad|>`, `<|unk|>`, `<|bos|>`, `<|eos|>`, `<|mask|>`, and basic chat tags.

### TARGET
- Complete structured chat and agent token suite:
  ```text
  <|system|>
  <|user|>
  <|assistant|>
  <|thought|> / <|thinking|>
  <|tool_call|>
  <|tool_result|>
  <|end|>
  ```
- Native Jinja2 chat templating engine conforming to standard conversational schemas.

### MIGRATION
1. Expand vocabulary configuration in [`src/model/vocabulary.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/model/vocabulary.py) preserving append-only index order.
2. Build verified chat formatter in [`src/inference/chat_session.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/inference/chat_session.py).
3. Verify backward compatibility with existing checkpoints via `tests/test_vocabulary_compatibility.py`.

---

## 4. Pretraining & Data Pipeline

### CURRENT
- Fixed mixture: WikiText-103 (90%) + TinyStories (10%).
- Basic sample-level deduplication and length filtering.

### TARGET
- Large-scale multi-domain dataset mixture:
  - High-quality Web text (FineWeb-Edu).
  - Code & algorithms (The Stack / StarCoder corpus).
  - Mathematics & scientific reasoning (OpenWebMath, Proof-Pile).
  - Structured instructional data.
- Scalable deduplication pipeline (MinHash LSH, exact duplicate removal).
- Automated PII and toxicity filtering with dataset governance audits.

### MIGRATION
1. Enhance [`src/datasets/filters.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/datasets/filters.py) with MinHash deduplication and quality scoring heuristics.
2. Configure weighted multi-domain data mixing in `configs/pretraining.gpu.yaml`.
3. Pre-shard cleaned data into binary token shards using `scripts/build_token_shards.py`.

---

## 5. Post-Training, Alignment & Reasoning

### CURRENT
- Basic supervised fine-tuning (SFT) scripts and early-stage Direct Preference Optimization (DPO) prototypes.

### TARGET
- Three-stage post-training pipeline:
  1. **Supervised Fine-Tuning (SFT)**: High-quality instruction-following dialogues, formatting consistency, and safety alignment.
  2. **Reasoning SFT**: Step-by-step chain-of-thought (CoT) prompting with `<thinking>...</thinking>` tags on math (GSM8K) and code problems.
  3. **Preference Optimization**: Stable Direct Preference Optimization (DPO) and Odds Ratio Preference Optimization (ORPO) for preference alignment and refusal of harmful requests.

### MIGRATION
1. Standardize instruction data formats in [`data/processed/`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/data/processed/).
2. Run SFT training with EMA weight tracking and plateau-based learning rate decay.
3. Validate zero-regression using retention test suite [`tests/test_retention_training.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/tests/test_retention_training.py).

---

## 6. Inference Optimization & Serving

### CURRENT
- Dynamic batching with basic micro-batch queue.
- Standard tensor KV cache and basic Paged KV cache implementation.
- Dynamic INT8 weight quantization.
- WebSocket streaming and basic FastAPI REST endpoints.

### TARGET
- **Continuous Batching & Paged KV Cache**:
  - vLLM-style virtual memory management for KV cache blocks.
  - Prefix caching to reuse system prompt and tool schema representations across requests.
- **Quantization**:
  - INT8 and INT4 weight-only and activation quantization (AWQ/GPTQ formats).
  - INT8 KV cache quantization to quadruple concurrency on 4 GB VRAM.
- **Speculative Decoding**:
  - Fast draft model generating candidate tokens verified in parallel by the target model.
- **OpenAI-Compatible REST API**:
  - Full `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/v1/embeddings`.

### MIGRATION
1. Integrate prefix caching into [`src/inference/paged_kv_cache.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/inference/paged_kv_cache.py).
2. Add OpenAI-compatible router in [`src/serving/api.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/serving/api.py).
3. Validate throughput and latency metrics using `src/evaluation/load_testing.py`.

---

## 7. Autonomous Agent & Tool Integration

### CURRENT
- MCP client (`src/mcp/client.py`) and basic local tool registry (`src/inference/local_tools.py`).
- Basic workspace file operations and search.

### TARGET
- Autonomous multi-step agent loop:
  - Plan $\rightarrow$ Select Tool $\rightarrow$ Execute $\rightarrow$ Observe $\rightarrow$ Reflect $\rightarrow$ Output.
  - Strict JSON schema validation and grammar-constrained decoding for tool arguments.
  - Integration with Model Context Protocol (MCP) servers for external tools (database, web browsing, IDE tools).
  - Retrieval-Augmented Generation (RAG) with local vector store and reranker.

### MIGRATION
1. Build structured JSON output enforcement in [`src/inference/generator.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/inference/generator.py).
2. Wire MCP client tool schemas directly into chat session system prompts.
3. Add end-to-end integration tests in `tests/test_workspace_agent.py`.

