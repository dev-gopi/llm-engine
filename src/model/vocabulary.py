"""Verified tokenizer/model vocabulary compatibility helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tokenizer.encoder import Tokenizer


THINKING_TOKENS = ("<thinking>", "</thinking>")


def extend_tokenizer_for_reasoning(tokenizer: Tokenizer) -> Tokenizer:
    """Create a verified append-only tokenizer with atomic reasoning tags."""
    missing = [token for token in THINKING_TOKENS if token not in tokenizer.vocab]
    return tokenizer.extend(missing) if missing else tokenizer


def adapt_config_to_tokenizer(
    model_config: Mapping[str, Any], tokenizer: Tokenizer
) -> dict[str, Any]:
    """Return a model config sized for an exact or append-only tokenizer."""
    config = dict(model_config)
    configured_size = int(config["vocab_size"])
    if tokenizer.vocab_size == configured_size:
        return config
    if (
        tokenizer.base_vocab_size == configured_size
        and tokenizer.vocab_size > configured_size
        and tokenizer.compatible_base_fingerprints
    ):
        config["vocab_size"] = tokenizer.vocab_size
        return config
    raise ValueError(
        f"tokenizer vocabulary ({tokenizer.vocab_size}) does not match model vocabulary "
        f"({configured_size}) and is not a verified append-only extension"
    )


def checkpoint_tokenizer_options(tokenizer: Tokenizer, *, allow_extension: bool = True) -> dict[str, Any]:
    """Arguments for loading either this tokenizer's checkpoint or its base."""
    return {
        "expected_tokenizer_fingerprint": tokenizer.fingerprint,
        "compatible_tokenizer_fingerprints": tokenizer.compatible_base_fingerprints if allow_extension else (),
        "allow_vocab_extension": allow_extension and bool(tokenizer.compatible_base_fingerprints),
    }


__all__ = [
    "THINKING_TOKENS",
    "adapt_config_to_tokenizer",
    "checkpoint_tokenizer_options",
    "extend_tokenizer_for_reasoning",
]
