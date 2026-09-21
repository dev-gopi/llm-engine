# Compact Context: Context Management (`docs/context/context-management.context.md`)

> **AGENT CONTEXT PACK**: Load this file when optimizing token budgets, pruning multi-turn conversations, managing RAG context injection, or configuring prefix caching.

---

## 1. Authoritative Sources of Truth
- **Context Compactor**: [`src/inference/context.py`](../../src/inference/context.py)
- **Chat Session**: [`src/inference/chat_session.py`](../../src/inference/chat_session.py)
- **Model Config**: [`configs/model.gpu.yaml`](../../configs/model.gpu.yaml) (`max_position: 1024`)

---

## 2. Token Budgeting Strategy
- Max context budget: `1024` tokens.
- System prompt & tools: `~200` tokens (pinned).
- Dialogue history & RAG: `~600` tokens (pruned FIFO).
- Generation headroom: `~200` tokens (reserved).

---

## 3. Key Invariants
1. Never truncate the system prompt or tool schemas.
2. Evict intermediate user/assistant turns when total tokens exceed `max_position - max_tokens`.
3. Paged KV cache prefix blocks are reused across turns if the system prompt has not changed.

---

## 4. Primary Verification Tests
```bash
.venv/bin/pytest tests/test_context.py tests/test_chat_session.py -q
```

