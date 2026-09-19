# Memory Systems & Session State (`docs/MEMORY.md`)

*Authoritative Source: [`src/inference/chat_session.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/inference/chat_session.py), [`src/serving/api.py`](file:///home/user/Downloads/llm-engine-boilerplate/llm-engine/src/serving/api.py)*

---

## 1. Memory Hierarchy

The engine implements a three-tier memory model:

```text
1. Working Memory (Active KV Cache)
   └── Volatile GPU tensors for the ongoing generation step.
2. Short-Term Episodic Memory (Session Context)
   └── Multi-turn conversation messages preserved in host RAM during a session.
3. Long-Term Persistent Memory (SQLite Session Store)
   └── Serialized conversation turns, user preferences, and metadata in data/cache/sessions.sqlite.
```

---

## 2. Session Persistence Architecture

- **Storage Engine**: SQLite with Write-Ahead Logging (WAL) enabled (`data/cache/sessions.sqlite`).
- **Data Model**:
  - `session_id` (UUID): Unique conversational session identifier.
  - `created_at` / `updated_at`: Timestamps.
  - `messages`: JSON array of turn objects (`role`, `content`, `tool_calls`, `tool_call_id`).
  - `metadata`: Generation settings, model checkpoint version, token usage totals.
- **Eviction Policy**: Configurable TTL (Time-To-Live) for stale inactive sessions to prevent unbounded disk growth.

