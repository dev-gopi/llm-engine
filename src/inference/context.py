"""Bounded in-memory conversation context for chat generation."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
import json
import sqlite3
import time
from pathlib import Path

from datasets.preprocessor import clean, format_messages
from tokenizer.encoder import Tokenizer


@dataclass(frozen=True)
class Message:
    role: str
    content: str


class ConversationMemory:
    """Keep recent messages within a tokenizer-measured context budget.

    This is process-local memory. Production multi-worker deployments should
    persist snapshots in a shared session store.
    """

    def __init__(self, tokenizer: Tokenizer, *, max_tokens: int, system_prompt: str | None = None) -> None:
        if max_tokens < 2:
            raise ValueError("max_tokens must be at least two")
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self._messages: list[Message] = []
        self._lock = RLock()
        if system_prompt:
            self._messages.append(Message("system", clean(system_prompt)))

    def add(self, role: str, content: str) -> None:
        normalized = clean(content)
        if role not in {"system", "user", "assistant"}:
            raise ValueError("role must be system, user, or assistant")
        if not normalized:
            raise ValueError("message content cannot be empty")
        with self._lock:
            self._messages.append(Message(role, normalized))
            self._trim()

    def render(self, *, add_generation_prompt: bool = True, reserve_tokens: int = 0) -> str:
        if reserve_tokens < 0:
            raise ValueError("reserve_tokens must be non-negative")
        with self._lock:
            self._trim(budget=self.max_tokens - reserve_tokens, add_generation_prompt=add_generation_prompt)
            return format_messages(
                [{"role": message.role, "content": message.content} for message in self._messages],
                add_generation_prompt=add_generation_prompt,
            )

    def snapshot(self) -> tuple[Message, ...]:
        with self._lock:
            return tuple(self._messages)

    def clear(self, *, preserve_system: bool = True) -> None:
        with self._lock:
            self._messages = [message for message in self._messages if preserve_system and message.role == "system"]

    def restore(self, messages: list[dict[str, str]]) -> None:
        with self._lock:
            self._messages = [Message(message["role"], clean(message["content"])) for message in messages]
            self._trim()

    def _trim(self, *, budget: int | None = None, add_generation_prompt: bool = False) -> None:
        budget = budget if budget is not None else self.max_tokens
        max_prompt_budget = max(1, self.max_tokens - 1)
        target_budget = max(1, min(budget, max_prompt_budget))
        while self._token_count(add_generation_prompt) > target_budget:
            non_system_indices = [index for index, message in enumerate(self._messages) if message.role != "system"]
            if not non_system_indices:
                raise ValueError("system prompt alone exceeds the context budget")
            if len(non_system_indices) == 1 and self._token_count(add_generation_prompt) <= max_prompt_budget:
                break
            self._messages.pop(non_system_indices[0])
        if self._token_count(add_generation_prompt) > max_prompt_budget:
            non_system_indices = [index for index, message in enumerate(self._messages) if message.role != "system"]
            if not non_system_indices:
                raise ValueError("system prompt alone exceeds the context budget")
            self._messages.pop(non_system_indices[0])
        if not self._messages:
            raise ValueError("conversation context is empty")

    def _token_count(self, add_generation_prompt: bool) -> int:
        if not self._messages:
            return 0
        text = format_messages(
            [{"role": message.role, "content": message.content} for message in self._messages],
            add_generation_prompt=add_generation_prompt,
        )
        return len(self.tokenizer.encode(text, add_bos=True, allowed_special="all"))


class SQLiteSessionStore:
    """Process-safe persistent conversation sessions backed by SQLite."""

    def __init__(self, path: str | Path, tokenizer: Tokenizer, *, max_tokens: int, system_prompt: str, ttl_seconds: int = 86400) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.ttl_seconds = ttl_seconds
        self._database_lock = RLock()
        with self._connect() as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, messages TEXT NOT NULL, updated REAL NOT NULL)")

    def load(self, session_id: str) -> ConversationMemory:
        memory = ConversationMemory(self.tokenizer, max_tokens=self.max_tokens, system_prompt=self.system_prompt)
        with self._database_lock, self._connect() as connection:
            row = connection.execute("SELECT messages, updated FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if row and row[1] >= time.time() - self.ttl_seconds:
            memory.restore(json.loads(row[0]))
        return memory

    def save(self, session_id: str, memory: ConversationMemory) -> None:
        payload = json.dumps([{"role": message.role, "content": message.content} for message in memory.snapshot()])
        with self._database_lock, self._connect() as connection:
            connection.execute(
                "INSERT INTO sessions VALUES (?, ?, ?) ON CONFLICT(id) DO UPDATE SET messages=excluded.messages, updated=excluded.updated",
                (session_id, payload, time.time()),
            )

    def delete(self, session_id: str) -> None:
        with self._database_lock, self._connect() as connection:
            connection.execute("DELETE FROM sessions WHERE id = ?", (session_id,))

    def _connect(self):
        return sqlite3.connect(self.path, timeout=5)
