"""Text and chat-record preprocessing."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any

VALID_ROLES = {"system", "user", "assistant"}


def clean(text: str) -> str:
    """Normalize Unicode/newlines and remove unsafe control characters."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    text = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(character for character in text if character in "\n\t" or unicodedata.category(character) != "Cc")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def format_messages(messages: Sequence[Mapping[str, Any]], *, add_generation_prompt: bool = False) -> str:
    """Serialize chat messages using the tokenizer's role special tokens."""
    if not messages:
        raise ValueError("messages cannot be empty")
    chunks: list[str] = []
    for message in messages:
        role = message.get("role")
        content = clean(message.get("content", ""))
        if role not in VALID_ROLES:
            raise ValueError(f"unsupported message role: {role!r}")
        if not content:
            raise ValueError("message content cannot be empty")
        chunks.append(f"<|{role}|>\n{content}\n")
    if add_generation_prompt:
        chunks.append("<|assistant|>\n")
    return "".join(chunks)


def record_to_text(record: Mapping[str, Any]) -> str:
    if isinstance(record.get("messages"), list):
        return format_messages(record["messages"])
    for key in ("text", "utterance", "content"):
        if isinstance(record.get(key), str):
            value = clean(record[key])
            if value:
                return value
    raise ValueError("record contains no supported text or messages field")
