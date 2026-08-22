"""Public byte-level BPE tokenizer API and artifact persistence."""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Collection, Iterable
from pathlib import Path
from typing import Any

import regex

from .bpe import BPE, BYTE_DECODER, BYTE_ENCODER


TOKENIZER_VERSION = 1
DEFAULT_PATTERN = (
    r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|"
    r"[^\r\n\p{L}\p{N}]?+\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|"
    r"\s*[\r\n]|\s+(?!\S)|\s+"
)
DEFAULT_SPECIAL_TOKENS = (
    "<|pad|>",
    "<|unk|>",
    "<|bos|>",
    "<|eos|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
)


class Tokenizer:
    """A reversible byte-level BPE tokenizer with explicit special-token handling."""

    def __init__(
        self,
        vocab: dict[str, int],
        merges: Iterable[tuple[str, str]] = (),
        *,
        special_tokens: dict[str, int] | None = None,
        pattern: str = DEFAULT_PATTERN,
        metadata: dict[str, Any] | None = None,
    ):
        if len(vocab) != len(set(vocab.values())):
            raise ValueError("vocabulary IDs must be unique")
        if set(vocab.values()) != set(range(len(vocab))):
            raise ValueError("vocabulary IDs must be contiguous and start at zero")

        self.vocab = dict(vocab)
        self.id_to_token = {identifier: token for token, identifier in vocab.items()}
        self.special_tokens = dict(special_tokens or {})
        for token, identifier in self.special_tokens.items():
            if self.vocab.get(token) != identifier:
                raise ValueError(f"special token {token!r} is missing from vocabulary")
        self.special_ids = {identifier: token for token, identifier in self.special_tokens.items()}
        self.pattern = pattern
        self._pattern = regex.compile(pattern)
        self.bpe = BPE(merges)
        self.metadata = dict(metadata or {})

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def token_to_id(self, token: str) -> int | None:
        return self.vocab.get(token)

    def id_to_piece(self, identifier: int) -> str | None:
        return self.id_to_token.get(identifier)

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
        allowed_special: Collection[str] | str = (),
    ) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        allowed = self._resolve_allowed_special(allowed_special)
        identifiers: list[int] = []
        if add_bos:
            identifiers.append(self._required_special_id("<|bos|>"))

        chunks = [text]
        if allowed:
            special_pattern = re.compile(
                "(" + "|".join(re.escape(token) for token in sorted(allowed, key=len, reverse=True)) + ")"
            )
            chunks = special_pattern.split(text)

        for chunk in chunks:
            if not chunk:
                continue
            if chunk in allowed:
                identifiers.append(self.special_tokens[chunk])
            else:
                identifiers.extend(self._encode_ordinary(chunk))

        if add_eos:
            identifiers.append(self._required_special_id("<|eos|>"))
        return identifiers

    def _encode_ordinary(self, text: str) -> list[int]:
        identifiers: list[int] = []
        for match in self._pattern.finditer(text):
            symbols = tuple(BYTE_ENCODER[value] for value in match.group(0).encode("utf-8"))
            for piece in self.bpe.apply(symbols):
                try:
                    identifiers.append(self.vocab[piece])
                except KeyError as error:
                    raise ValueError(f"BPE piece is absent from vocabulary: {piece!r}") from error
        return identifiers

    def decode(self, identifiers: Iterable[int], *, skip_special_tokens: bool = False) -> str:
        output: list[str] = []
        byte_buffer = bytearray()

        def flush_bytes() -> None:
            if byte_buffer:
                output.append(byte_buffer.decode("utf-8", errors="replace"))
                byte_buffer.clear()

        for raw_identifier in identifiers:
            identifier = int(raw_identifier)
            if identifier not in self.id_to_token:
                raise ValueError(f"token ID is outside the vocabulary: {identifier}")
            if identifier in self.special_ids:
                flush_bytes()
                if not skip_special_tokens:
                    output.append(self.special_ids[identifier])
                continue
            piece = self.id_to_token[identifier]
            try:
                byte_buffer.extend(BYTE_DECODER[character] for character in piece)
            except KeyError as error:
                raise ValueError(f"token contains an invalid byte symbol: {piece!r}") from error
        flush_bytes()
        return "".join(output)

    def save(self, directory: str | Path) -> Path:
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        artifact = destination / "tokenizer.json"
        payload = {
            "version": TOKENIZER_VERSION,
            "type": "byte_level_bpe",
            "pattern": self.pattern,
            "vocab": self.vocab,
            "merges": [list(pair) for pair in self.bpe.merges],
            "special_tokens": self.special_tokens,
            "metadata": self.metadata,
        }
        _atomic_json_dump(artifact, payload)
        _atomic_json_dump(destination / "vocab.json", self.vocab)
        _atomic_text_write(
            destination / "merges.txt",
            "#version: 1\n" + "".join(f"{left} {right}\n" for left, right in self.bpe.merges),
        )
        return artifact

    @classmethod
    def load(cls, path: str | Path) -> "Tokenizer":
        artifact = Path(path)
        if artifact.is_dir():
            artifact = artifact / "tokenizer.json"
        with artifact.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        if payload.get("version") != TOKENIZER_VERSION:
            raise ValueError(f"unsupported tokenizer version: {payload.get('version')!r}")
        if payload.get("type") != "byte_level_bpe":
            raise ValueError(f"unsupported tokenizer type: {payload.get('type')!r}")
        return cls(
            payload["vocab"],
            (tuple(pair) for pair in payload["merges"]),
            special_tokens=payload["special_tokens"],
            pattern=payload["pattern"],
            metadata=payload.get("metadata"),
        )

    def _resolve_allowed_special(self, allowed: Collection[str] | str) -> set[str]:
        if allowed == "all":
            return set(self.special_tokens)
        if isinstance(allowed, str):
            raise ValueError("allowed_special must be a collection or 'all'")
        result = set(allowed)
        unknown = result.difference(self.special_tokens)
        if unknown:
            raise ValueError(f"unknown special tokens: {sorted(unknown)}")
        return result

    def _required_special_id(self, token: str) -> int:
        try:
            return self.special_tokens[token]
        except KeyError as error:
            raise ValueError(f"tokenizer does not define required special token {token!r}") from error


def _atomic_json_dump(path: Path, payload: object) -> None:
    _atomic_text_write(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _atomic_text_write(path: Path, content: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
