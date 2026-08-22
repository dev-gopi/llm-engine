"""Validated container for per-layer Transformer key/value caches."""

from __future__ import annotations

import torch
from torch import Tensor

LayerCache = tuple[Tensor, Tensor]


class KVCache:
    def __init__(self, values: tuple[LayerCache, ...] = ()) -> None:
        self.values = values
        self._validate()

    @property
    def sequence_length(self) -> int:
        return self.values[0][0].shape[2] if self.values else 0

    def update(self, values: tuple[LayerCache, ...]) -> None:
        self.values = values
        self._validate()

    def clear(self) -> None:
        self.values = ()

    def crop(self, max_length: int) -> None:
        if max_length < 0:
            raise ValueError("max_length must be non-negative")
        self.values = tuple((key[:, :, -max_length:], value[:, :, -max_length:]) for key, value in self.values) if max_length else ()

    def to(self, device: str | torch.device) -> "KVCache":
        return KVCache(tuple((key.to(device), value.to(device)) for key, value in self.values))

    def _validate(self) -> None:
        length = None
        for key, value in self.values:
            if key.ndim != 4 or key.shape != value.shape:
                raise ValueError("cached keys and values must have equal four-dimensional shapes")
            if length is not None and key.shape[2] != length:
                raise ValueError("all layer caches must have the same sequence length")
            length = key.shape[2]
