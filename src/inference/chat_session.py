"""Opt-in persistent chat and explicitly approved fine-tuning examples."""
from __future__ import annotations

from contextlib import closing
import json
from pathlib import Path
import sqlite3
from threading import RLock

import torch

from inference.context import SQLiteSessionStore


_CHAT_ROLES = frozenset({"system", "user", "assistant"})


def format_chat_messages(messages, *, add_generation_prompt: bool = False) -> str:
    """Render the canonical, unambiguous instruction-tuning chat format."""
    rendered: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("each message must be a mapping")
        role = message.get("role")
        content = message.get("content")
        if role not in _CHAT_ROLES:
            raise ValueError("message role must be system, user, or assistant")
        if not isinstance(content, str):
            raise ValueError("message content must be text")
        rendered.append(f"<|{role}|>\n{content}\n<|end|>\n")
    if add_generation_prompt:
        rendered.append("<|assistant|>\n")
    return "".join(rendered)


def build_chat_sft_example(tokenizer, messages) -> dict[str, torch.Tensor]:
    """Tokenize chat turns and supervise only assistant responses and end tags."""
    token_ids: list[int] = []
    loss_mask: list[bool] = []
    for index, message in enumerate(messages):
        # Validate all records before tokenizing so malformed role labels can
        # never silently become supervised training data.
        format_chat_messages([message])
        role = message["role"]
        header = tokenizer.encode(
            f"<|{role}|>\n", add_bos=index == 0, allowed_special="all"
        )
        content = tokenizer.encode(message["content"], add_bos=False)
        end = tokenizer.encode("\n<|end|>\n", add_bos=False, allowed_special="all")
        token_ids.extend(header)
        token_ids.extend(content)
        token_ids.extend(end)
        supervise = role == "assistant"
        loss_mask.extend([False] * len(header))
        loss_mask.extend([supervise] * len(content))
        loss_mask.extend([supervise] * len(end))
    if not token_ids:
        raise ValueError("at least one chat message is required")
    inputs = torch.tensor(token_ids, dtype=torch.long)
    mask = torch.tensor(loss_mask, dtype=torch.bool)
    labels = inputs.masked_fill(~mask, -100)
    return {"input_ids": inputs, "labels": labels, "loss_mask": mask}


class ChatSession:
    """Single-process session handle; serialize use of its owning backend."""

    def __init__(self, backend, path, session_id, *, system_prompt="", ttl_seconds=86400):
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be nonempty")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        self.backend = backend
        self.session_id = session_id
        self.store = SQLiteSessionStore(path, backend.tokenizer,
                                       max_tokens=backend.generator.max_positions - 1,
                                       system_prompt=system_prompt, ttl_seconds=ttl_seconds)
        self._lock = RLock()
        self._pending = None
        with closing(sqlite3.connect(self.store.path)) as connection, connection:
            connection.execute("CREATE TABLE IF NOT EXISTS approved_chat_examples "
                               "(session_id TEXT NOT NULL, messages TEXT NOT NULL, "
                               "PRIMARY KEY(session_id, messages))")

    def chat(self, message, **options):
        """Persist a successful turn. Training export remains separately opt-in."""
        with self._lock:
            self._pending = None
            memory = self.store.load(self.session_id)
            memory.add("user", message)
            settings = self.backend.generation_config | options
            maximum = int(settings.get("max_tokens", 128))
            prompt = memory.render(reserve_tokens=maximum)
            messages = [{"role": item.role, "content": item.content} for item in memory.snapshot()]
            result = self.backend.generate(prompt, **(settings | {"allow_special_tokens": True}))
            if result.text.strip():
                # Capture exactly the bounded prompt used, before history trims.
                self._pending = messages + [{"role": "assistant", "content": result.text}]
                memory.add("assistant", result.text)
                self.store.save(self.session_id, memory)
            return result

    def history(self):
        with self._lock:
            return [{"role": item.role, "content": item.content}
                    for item in self.store.load(self.session_id).snapshot()]

    def review_pending(self):
        """Return a copy of the exact bounded conversation awaiting approval."""
        with self._lock:
            return [dict(item) for item in (self._pending or [])]

    def approve_conversation(self, *, corrected_response=None):
        """Approve the entire last prompt/reply snapshot, including prior replies."""
        with self._lock:
            if self._pending is None:
                raise ValueError("no unreviewed response; generate a successful reply first")
            messages = [dict(item) for item in self._pending]
            if corrected_response is not None:
                if not isinstance(corrected_response, str) or not corrected_response.strip():
                    raise ValueError("corrected_response must be nonempty text")
                messages[-1]["content"] = corrected_response
            payload = json.dumps(messages, ensure_ascii=False, sort_keys=True)
            with closing(sqlite3.connect(self.store.path)) as connection, connection:
                connection.execute("INSERT OR IGNORE INTO approved_chat_examples VALUES (?, ?)",
                                   (self.session_id, payload))
            self._pending = None

    def export_training(self, path):
        """Export only this session's approved examples; never overwrite a file."""
        with self._lock:
            with closing(sqlite3.connect(self.store.path)) as connection:
                rows = connection.execute(
                    "SELECT messages FROM approved_chat_examples WHERE session_id = ? ORDER BY rowid",
                    (self.session_id,)).fetchall()
            if not rows:
                raise ValueError("no approved examples to export")
            destination = Path(path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("x", encoding="utf-8") as stream:
                for (messages,) in rows:
                    stream.write(json.dumps({"messages": json.loads(messages)}, ensure_ascii=False) + "\n")
            return len(rows)

    def forget(self, *, include_training_examples=False):
        """Delete stored history; optionally delete this session's approved data."""
        with self._lock:
            self.store.delete(self.session_id)
            self._pending = None
            if include_training_examples:
                with closing(sqlite3.connect(self.store.path)) as connection, connection:
                    connection.execute("DELETE FROM approved_chat_examples WHERE session_id = ?",
                                       (self.session_id,))
