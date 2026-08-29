"""Validated container for per-layer Transformer key/value caches."""

from __future__ import annotations

import torch
from torch import Tensor

from model.kv_cache import StaticLayerKVCache

LayerCache = tuple[Tensor, Tensor]


class KVCache:
    def __init__(self, values: tuple[LayerCache, ...] = (), *, capacity: int | None = None) -> None:
        self.values = (
            tuple(StaticLayerKVCache(key, value, capacity=capacity) for key, value in values)
            if capacity is not None else values
        )
        self._validate()

    @property
    def sequence_length(self) -> int:
        if not self.values:
            return 0
        first = self.values[0]
        return first.length if isinstance(first, StaticLayerKVCache) else first[0].shape[2]

    def update(self, values) -> None:
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
        for layer in self.values:
            if isinstance(layer, StaticLayerKVCache):
                current = layer.length
                if length is not None and current != length:
                    raise ValueError("all layer caches must have the same sequence length")
                length = current
                continue
            key, value = layer
            if key.ndim != 4 or key.shape != value.shape:
                raise ValueError("cached keys and values must have equal four-dimensional shapes")
            if length is not None and key.shape[2] != length:
                raise ValueError("all layer caches must have the same sequence length")
            length = key.shape[2]
