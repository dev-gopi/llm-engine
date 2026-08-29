"""Verified tokenizer/model vocabulary compatibility helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tokenizer.encoder import Tokenizer


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


def checkpoint_tokenizer_options(tokenizer: Tokenizer) -> dict[str, Any]:
    """Arguments for loading either this tokenizer's checkpoint or its base."""
    return {
        "expected_tokenizer_fingerprint": tokenizer.fingerprint,
        "compatible_tokenizer_fingerprints": tokenizer.compatible_base_fingerprints,
        "allow_vocab_extension": bool(tokenizer.compatible_base_fingerprints),
    }


__all__ = ["adapt_config_to_tokenizer", "checkpoint_tokenizer_options"]
