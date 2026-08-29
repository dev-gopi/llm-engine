"""Public byte-level BPE tokenizer API and artifact persistence."""

from __future__ import annotations

import json
import hashlib
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

    @property
    def fingerprint(self) -> str:
        """Stable identity for detecting incompatible same-size vocabularies."""
        payload = {
            "pattern": self.pattern,
            "vocab": sorted(self.vocab.items(), key=lambda item: item[1]),
            "merges": list(self.bpe.merges),
            "special_tokens": sorted(self.special_tokens.items()),
        }
        if self.added_tokens:
            payload["added_tokens"] = list(self.added_tokens)
        encoded = json.dumps(
            payload, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @property
    def compatible_base_fingerprints(self) -> frozenset[str]:
        """Tokenizer fingerprints whose ID mappings are preserved as a prefix."""
        extension = self.metadata.get("extension", {})
        ancestors = extension.get("compatible_base_fingerprints", ())
        if not isinstance(ancestors, list) or not all(isinstance(item, str) for item in ancestors):
            return frozenset()
        return frozenset(ancestors)

    @property
    def base_vocab_size(self) -> int | None:
        extension = self.metadata.get("extension", {})
        value = extension.get("base_vocab_size")
        return value if isinstance(value, int) and value > 0 else None

    @property
    def added_tokens(self) -> tuple[str, ...]:
        extension = self.metadata.get("extension", {})
        values = extension.get("added_token_texts", ())
        if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
            return ()
        return tuple(values)

    def extend(self, tokens: Iterable[str]) -> "Tokenizer":
        """Return an append-only BPE extension while preserving every existing ID.

        Added tokens are matched explicitly before ordinary regex/BPE encoding.
        This supports scripts with combining marks, emoji sequences, phrases,
        and code literals without changing existing merge priorities.
        """
        vocab = dict(self.vocab)
        added_texts = list(self.added_tokens)
        added_pieces: list[str] = []

        for text in tokens:
            if not isinstance(text, str) or not text:
                raise ValueError("extension tokens must be non-empty strings")
            if text in self.special_tokens:
                raise ValueError(f"special token cannot be added as an ordinary token: {text!r}")
            if text in added_texts:
                continue
            piece = "".join(BYTE_ENCODER[value] for value in text.encode("utf-8"))
            if piece not in vocab:
                vocab[piece] = len(vocab)
                added_pieces.append(piece)
            added_texts.append(text)

        ancestors = [self.fingerprint, *sorted(self.compatible_base_fingerprints)]
        metadata = dict(self.metadata)
        metadata["extension"] = {
            "base_fingerprint": self.fingerprint,
            "base_vocab_size": self.base_vocab_size or self.vocab_size,
            "parent_vocab_size": self.vocab_size,
            "compatible_base_fingerprints": list(dict.fromkeys(ancestors)),
            "added_vocab_size": len(vocab) - self.vocab_size,
            "added_tokens": added_pieces,
            "added_token_texts": added_texts,
        }
        return Tokenizer(
            vocab,
            self.bpe.merges,
            special_tokens=self.special_tokens,
            pattern=self.pattern,
            metadata=metadata,
        )

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
        chunks = [text]
        if self.added_tokens:
            added_pattern = re.compile(
                "(" + "|".join(
                    re.escape(token) for token in sorted(self.added_tokens, key=len, reverse=True)
                ) + ")"
            )
            chunks = added_pattern.split(text)
        for chunk in chunks:
            if not chunk:
                continue
            if chunk in self.added_tokens:
                piece = "".join(BYTE_ENCODER[value] for value in chunk.encode("utf-8"))
                identifiers.append(self.vocab[piece])
                continue
            identifiers.extend(self._encode_bpe_chunk(chunk))
        return identifiers

    def _encode_bpe_chunk(self, text: str) -> list[int]:
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
            "fingerprint": self.fingerprint,
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
        tokenizer = cls(
            payload["vocab"],
            (tuple(pair) for pair in payload["merges"]),
            special_tokens=payload["special_tokens"],
            pattern=payload["pattern"],
            metadata=payload.get("metadata"),
        )
        expected_fingerprint = payload.get("fingerprint")
        if expected_fingerprint is not None and expected_fingerprint != tokenizer.fingerprint:
            raise ValueError("tokenizer artifact fingerprint does not match its contents")
        return tokenizer

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
