"""Preallocated key/value cache primitives for incremental decoding."""

from __future__ import annotations

import torch
from torch import Tensor


class StaticLayerKVCache:
    """Fixed-capacity cache that appends new K/V states without concatenation."""

    def __init__(self, key: Tensor, value: Tensor, *, capacity: int) -> None:
        if key.ndim != 4 or key.shape != value.shape:
            raise ValueError("initial keys and values must have equal four-dimensional shapes")
        if capacity < key.shape[2]:
            raise ValueError("cache capacity cannot be smaller than its initial sequence")
        shape = (*key.shape[:2], capacity, key.shape[3])
        self.key = key.new_empty(shape)
        self.value = value.new_empty(shape)
        self.length = key.shape[2]
        self.key[:, :, : self.length].copy_(key)
        self.value[:, :, : self.length].copy_(value)

    def append(self, key: Tensor, value: Tensor) -> tuple[Tensor, Tensor]:
        if key.ndim != 4 or key.shape != value.shape:
            raise ValueError("appended keys and values must have equal four-dimensional shapes")
        if key.shape[:2] != self.key.shape[:2] or key.shape[3] != self.key.shape[3]:
            raise ValueError("appended keys and values do not match cache dimensions")
        stop = self.length + key.shape[2]
        if stop > self.key.shape[2]:
            raise ValueError("key/value cache capacity exceeded")
        self.key[:, :, self.length:stop].copy_(key)
        self.value[:, :, self.length:stop].copy_(value)
        self.length = stop
        return self.current()

    def current(self) -> tuple[Tensor, Tensor]:
        return self.key[:, :, : self.length], self.value[:, :, : self.length]

    def __getitem__(self, index: int) -> Tensor:
        return self.current()[index]

    def __iter__(self):
        return iter(self.current())

    def __len__(self) -> int:
        return 2


__all__ = ["StaticLayerKVCache"]
