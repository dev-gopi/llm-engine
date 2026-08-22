"""Decoding interface for applications that only need token IDs to text."""

from __future__ import annotations

from collections.abc import Iterable

from .encoder import Tokenizer


class Decoder:
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def decode(self, identifiers: Iterable[int], *, skip_special_tokens: bool = False) -> str:
        return self.tokenizer.decode(identifiers, skip_special_tokens=skip_special_tokens)
