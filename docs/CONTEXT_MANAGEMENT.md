# Context Management & Token Budgets (`docs/CONTEXT_MANAGEMENT.md`)

*Authoritative Source: [`src/inference/context.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/inference/context.py), [`src/inference/chat_session.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/inference/chat_session.py)*

---

## 1. The Context Budget Invariant

With an active context length of `max_position: 1024` tokens, allocating token capacity dynamically is critical to prevent prompt overflow errors:

```text
┌────────────────────────────────────────────────────────┐
│ Total Context Budget: 1,024 Tokens                     │
├─────────────────────────┬──────────────────────────────┤
│ System Prompt & Tools   │ ~150 - 250 tokens            │
│ Retrieved RAG Context   │ ~200 - 300 tokens            │
│ Multi-Turn History      │ ~250 - 350 tokens            │
│ Generation Headroom     │ ~200 tokens (reserved)       │
└─────────────────────────┴──────────────────────────────┘
```

---

## 2. Conversation Compaction Strategies

When conversational turns approach the maximum context window, [`src/inference/context.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/inference/context.py) applies progressive compaction:

1. **System Prompt Protection**: System instructions, persona guidelines, and active tool schemas are pinned and never truncated.
2. **Oldest Turn Pruning**: Intermediate user/assistant conversation pairs are evicted first (FIFO).
3. **Observation Summarization**: Lengthy `<tool_result>` payloads (e.g. large file contents) are truncated or summarized into concise status messages.
4. **Sliding Window Caching**: The most recent $K$ turns are preserved intact to maintain immediate conversational coherence.

---

## 3. Prefix Caching for Long System Prompts

In multi-agent and tool-calling setups, system prompts and JSON tool declarations remain static across requests:
- By caching the Key and Value representations of the invariant system prefix in [`src/inference/paged_kv_cache.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/inference/paged_kv_cache.py), new requests skip prefilling the first ~250 tokens.
- This reduces Time-To-First-Token (TTFT) by up to 70% for multi-turn interactions.

