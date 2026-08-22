"""Token embedding matrix used by the GPT model."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor


class TokenEmbedding(nn.Module):
    """Map token IDs to dense vectors with GPT-compatible initialization.

    The module owns only the token embedding matrix. Position embeddings and
    embedding dropout remain separate model components. Vocabulary resizing
    should be completed before constructing an optimizer because it replaces
    the underlying parameter object.
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        *,
        padding_idx: int | None = None,
        initializer_range: float = 0.02,
        scale_embeddings: bool = False,
        freeze: bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self._validate_configuration(vocab_size, dim, padding_idx, initializer_range)

        self.initializer_range = float(initializer_range)
        self.scale_embeddings = bool(scale_embeddings)
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            padding_idx=padding_idx,
            device=device,
            dtype=dtype,
        )
        self.reset_parameters()
        self.embedding.weight.requires_grad_(not freeze)

    @property
    def weight(self) -> nn.Parameter:
        return self.embedding.weight

    @property
    def vocab_size(self) -> int:
        return self.embedding.num_embeddings

    @property
    def embedding_dim(self) -> int:
        return self.embedding.embedding_dim

    @property
    def padding_idx(self) -> int | None:
        return self.embedding.padding_idx

    def reset_parameters(self) -> None:
        """Initialize weights and force the padding row to remain zero."""

        nn.init.normal_(self.embedding.weight, mean=0.0, std=self.initializer_range)
        self._zero_padding_row()

    def forward(self, token_ids: Tensor) -> Tensor:
        if not isinstance(token_ids, Tensor):
            raise TypeError("token_ids must be a torch.Tensor")
        if token_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("token_ids must use torch.int32 or torch.int64")
        embeddings = self.embedding(token_ids)
        if self.scale_embeddings:
            embeddings = embeddings * math.sqrt(self.embedding_dim)
        return embeddings

    def resize(self, new_vocab_size: int, *, pad_to_multiple_of: int | None = None) -> int:
        """Resize the vocabulary and retain every overlapping token vector.

        Returns the effective size, which may be larger than ``new_vocab_size``
        when ``pad_to_multiple_of`` is used for hardware-friendly dimensions.
        Any output projection tied before resizing must be tied again afterward.
        """

        if not isinstance(new_vocab_size, int) or isinstance(new_vocab_size, bool):
            raise TypeError("new_vocab_size must be an integer")
        if new_vocab_size < 1:
            raise ValueError("new_vocab_size must be positive")
        if pad_to_multiple_of is not None:
            if not isinstance(pad_to_multiple_of, int) or isinstance(pad_to_multiple_of, bool):
                raise TypeError("pad_to_multiple_of must be an integer")
            if pad_to_multiple_of < 1:
                raise ValueError("pad_to_multiple_of must be positive")
            new_vocab_size = math.ceil(new_vocab_size / pad_to_multiple_of) * pad_to_multiple_of
        if self.padding_idx is not None and self.padding_idx >= new_vocab_size:
            raise ValueError("new vocabulary would remove the configured padding token")
        if new_vocab_size == self.vocab_size:
            return new_vocab_size

        old_embedding = self.embedding
        new_embedding = nn.Embedding(
            num_embeddings=new_vocab_size,
            embedding_dim=self.embedding_dim,
            padding_idx=self.padding_idx,
            device=old_embedding.weight.device,
            dtype=old_embedding.weight.dtype,
        )
        nn.init.normal_(new_embedding.weight, mean=0.0, std=self.initializer_range)
        rows_to_copy = min(self.vocab_size, new_vocab_size)
        with torch.no_grad():
            new_embedding.weight[:rows_to_copy].copy_(old_embedding.weight[:rows_to_copy])
            if self.padding_idx is not None:
                new_embedding.weight[self.padding_idx].zero_()
        new_embedding.weight.requires_grad_(old_embedding.weight.requires_grad)
        self.embedding = new_embedding
        return new_vocab_size

    def tie_weights(self, output_projection: nn.Linear) -> None:
        """Share this matrix with a vocabulary-sized output projection."""

        if not isinstance(output_projection, nn.Linear):
            raise TypeError("output_projection must be torch.nn.Linear")
        expected_shape = (self.vocab_size, self.embedding_dim)
        if tuple(output_projection.weight.shape) != expected_shape:
            raise ValueError(
                "output projection weight shape must be "
                f"{expected_shape}, got {tuple(output_projection.weight.shape)}"
            )
        output_projection.weight = self.embedding.weight

    def freeze(self) -> "TokenEmbedding":
        self.embedding.weight.requires_grad_(False)
        return self

    def unfreeze(self) -> "TokenEmbedding":
        self.embedding.weight.requires_grad_(True)
        return self

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "TokenEmbedding":
        """Construct from the model YAML mapping without coupling to YAML I/O."""

        return cls(
            vocab_size=int(config["vocab_size"]),
            dim=int(config["hidden_size"]),
            padding_idx=config.get("padding_idx"),
            initializer_range=float(config.get("initializer_range", 0.02)),
            scale_embeddings=bool(config.get("scale_embeddings", False)),
            freeze=bool(config.get("freeze_embeddings", False)),
            device=device,
            dtype=dtype,
        )

    def extra_repr(self) -> str:
        return (
            f"vocab_size={self.vocab_size}, embedding_dim={self.embedding_dim}, "
            f"padding_idx={self.padding_idx}, initializer_range={self.initializer_range}, "
            f"scale_embeddings={self.scale_embeddings}"
        )

    @staticmethod
    def _validate_configuration(
        vocab_size: int,
        dim: int,
        padding_idx: int | None,
        initializer_range: float,
    ) -> None:
        for name, value in (("vocab_size", vocab_size), ("dim", dim)):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if padding_idx is not None:
            if not isinstance(padding_idx, int) or isinstance(padding_idx, bool):
                raise TypeError("padding_idx must be an integer or None")
            if not 0 <= padding_idx < vocab_size:
                raise ValueError("padding_idx must be inside the vocabulary")
        if not math.isfinite(initializer_range) or initializer_range <= 0:
            raise ValueError("initializer_range must be finite and positive")

    def _zero_padding_row(self) -> None:
        if self.padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[self.padding_idx].zero_()
