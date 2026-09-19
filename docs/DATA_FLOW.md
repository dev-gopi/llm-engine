# Data Flow & Tensor Transformations (`docs/DATA_FLOW.md`)

This document traces the data flow and tensor shape transformations across all major lifecycles in `llm-engine`.

---

## 1. Data Ingestion & Preprocessing Flow

```text
Raw Text (WikiText, TinyStories, Web)
        │
        ▼ (src/datasets/filters.py)
Quality & Governance Filtering (deduplication, PII redaction, min/max length)
        │
        ▼ (src/tokenizer/bpe.py)
Tokenization (Byte-Pair Encoding regex + merges)
        │ String -> List[int] (Token IDs)
        ▼ (scripts/build_token_shards.py)
Binary Sharding (packed uint16/uint32 arrays stored on disk)
        │
        ▼ (src/datasets/loader.py)
Streaming Dataloader with memmap & dynamic mixture sampling
        │ Batched Tensor: input_ids [B, T], labels [B, T]
        ▼
PyTorch Model Input
```

---

## 2. Training Step Forward & Backward Tensor Flow

For batch size $B=2$, sequence length $T=512$, hidden size $D=512$, vocab size $V=40,000$:

```text
1. Input IDs: [B, T] (torch.int64)
   │
   ▼ Token Embedding (src/model/embedding.py)
2. Hidden States: [B, T, D] = [2, 512, 512]
   │
   ▼ Transformer Blocks (x16 layers)
   │ ┌────────────────────────────────────────────────────────┐
   │ │ Layer Input: [B, T, D]                                 │
   │ │   │                                                    │
   │ │   ├── RMSNorm: [B, T, D]                               │
   │ │   ├── Q [B, H_q, T, D_h] = [2, 8, 512, 64]             │
   │ │   ├── K [B, H_kv, T, D_h] = [2, 2, 512, 64]            │
   │ │   ├── V [B, H_kv, T, D_h] = [2, 2, 512, 64]            │
   │ │   ├── RoPE: rotates Q and K in-place                   │
   │ │   ├── Repeat KV heads to match Q (2 -> 8 heads)        │
   │ │   ├── Attention: Softmax(Q @ K.T / sqrt(D_h)) @ V      │
   │ │   ├── Output Projection: [B, T, D]                     │
   │ │   └── Residual Addition: x = x + Attention_Out         │
   │ │                                                        │
   │ │   ├── RMSNorm: [B, T, D]                               │
   │ │   ├── SwiGLU FFN: (x @ W_gate * SiLU) * (x @ W_up)     │
   │ │   │   FFN Hidden: [B, T, 2048]                         │
   │ │   ├── FFN Down Projection: [B, T, D]                   │
   │ │   └── Residual Addition: x = x + FFN_Out               │
   │ └────────────────────────────────────────────────────────┘
   │
   ▼ Final RMSNorm (src/model/layer_norm.py)
3. Normalized Hidden States: [B, T, D]
   │
   ▼ LM Head (Weight Tied with Token Embedding)
4. Chunked Loss Computation (src/model/loss.py)
   │ Instead of allocating full [B, T, V] = [2, 512, 40000] logits:
   │ Loops over sequence chunks of size C=128:
   │   chunk_logits = hidden_chunk @ embedding_weight.T -> [B, C, V]
   │   chunk_loss = CrossEntropy(chunk_logits, labels_chunk)
   │   accumulate scalar loss
   ▼
5. Scalar Loss (with z-loss penalty)
   │
   ▼ Loss.backward() via PyTorch AMP GradScaler
6. Gradients computed with activation recomputation (Gradient Checkpointing)
```

---

## 3. Autoregressive Inference & KV Cache Flow

During inference generation:

```text
Prompt Tokens: [1, P]
        │
        ▼ Prefill Phase
Compute initial hidden states for all P prompt tokens.
Fill KV Cache: Key Cache [1, 2, P, 64], Value Cache [1, 2, P, 64].
Generate 1st new token: t_1.
        │
        ▼ Decode Phase (Step-by-Step for token t_k)
Input: [1, 1] (single token ID)
        │
        ├── Q: [1, 8, 1, 64]
        ├── New K: [1, 2, 1, 64]
        ├── New V: [1, 2, 1, 64]
        │
        ├── Append New K/V to Paged KV Cache:
        │   Total Cached Length: L = P + k
        │   Cached K: [1, 2, L, 64]
        │   Cached V: [1, 2, L, 64]
        │
        ├── Attention computed between Q [1, 8, 1, 64] and Cached K [1, 8, L, 64]
        │
        ▼ Logits for token t_k: [1, 1, V]
Apply Temperature, Top-K, Top-P filter (src/inference/sampler.py)
Sample next token t_{k+1} -> Stream to WebSocket
```

---

## 4. Agent & Tool Execution Flow

```text
User Message ("Check weather in Kolkata")
        │
        ▼ ChatSession (src/inference/chat_session.py)
Construct system prompt with tool JSON schemas
        │
        ▼ Generator (src/inference/generator.py)
Model emits structured tool call:
<tool_call>{"name": "get_weather", "arguments": {"city": "Kolkata"}}</tool_call>
        │
        ▼ Workspace Orchestrator (src/serving/workspace.py)
Parse JSON schema & validate arguments
        │
        ▼ MCP Client / Local Tool Runner (src/mcp/client.py)
Execute tool locally or remotely -> Result: {"temp": 28, "unit": "C"}
        │
        ▼ Context Injection
Append <tool_result>{"temp": 28, "unit": "C"}</tool_result> to conversation history
        │
        ▼ Generator (Second Pass)
Model consumes tool result and generates final human-readable response
```

