"""Deterministic, auditable corpus filters for large-scale preparation."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass


_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_PHONE = re.compile(r"(?<!\w)(?:\+?\d[\d .()-]{7,}\d)(?!\w)")
_IP = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


@dataclass
class FilterStats:
    accepted: int = 0
    empty: int = 0
    too_short: int = 0
    too_long: int = 0
    low_quality: int = 0
    language: int = 0
    duplicate: int = 0
    pii_redactions: int = 0


class CorpusFilter:
    def __init__(
        self,
        *,
        min_chars: int = 40,
        max_chars: int = 100_000,
        english_only: bool = False,
        redact_pii: bool = True,
    ) -> None:
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.english_only = english_only
        self.redact_pii = redact_pii
        self.seen: set[bytes] = set()
        self.stats = FilterStats()

    def apply(self, text: str) -> str | None:
        text = " ".join(str(text).split())
        if not text:
            self.stats.empty += 1
            return None
        if len(text) < self.min_chars:
            self.stats.too_short += 1
            return None
        if len(text) > self.max_chars:
            self.stats.too_long += 1
            return None
        printable = sum(character.isprintable() for character in text) / len(text)
        alphanumeric = sum(character.isalnum() for character in text) / len(text)
        if printable < 0.95 or alphanumeric < 0.25:
            self.stats.low_quality += 1
            return None
        if self.english_only:
            letters = [character for character in text if character.isalpha()]
            ascii_ratio = sum(character.isascii() for character in letters) / max(len(letters), 1)
            if ascii_ratio < 0.85:
                self.stats.language += 1
                return None
        digest = hashlib.blake2b(text.casefold().encode("utf-8"), digest_size=16).digest()
        if digest in self.seen:
            self.stats.duplicate += 1
            return None
        self.seen.add(digest)
        if self.redact_pii:
            text, count_email = _EMAIL.subn("<email>", text)
            text, count_phone = _PHONE.subn("<phone>", text)
            text, count_ip = _IP.subn("<ip>", text)
            self.stats.pii_redactions += count_email + count_phone + count_ip
        self.stats.accepted += 1
        return text
