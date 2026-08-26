"""Fixed-size page allocator for KV tensors used by batched serving engines."""

from __future__ import annotations

from collections import OrderedDict

import torch
from torch import Tensor


class PagedKVCache:
    def __init__(
        self,
        *,
        num_pages: int,
        page_size: int,
        layers: int,
        kv_heads: int,
        head_dim: int,
        device: str | torch.device,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        if min(num_pages, page_size, layers, kv_heads, head_dim) < 1:
            raise ValueError("paged cache dimensions must be positive")
        shape = (num_pages, layers, 2, kv_heads, page_size, head_dim)
        self.storage = torch.empty(shape, device=device, dtype=dtype)
        self.page_size = page_size
        self.free_pages = list(range(num_pages - 1, -1, -1))
        self.tables: dict[str, list[int]] = {}
        self.lengths: dict[str, int] = {}

    def reserve(self, request_id: str, token_capacity: int) -> None:
        if not request_id:
            raise ValueError("request_id cannot be empty")
        if not isinstance(token_capacity, int) or isinstance(token_capacity, bool) or token_capacity < 1:
            raise ValueError("token_capacity must be a positive integer")
        if request_id in self.tables:
            raise ValueError(f"request already exists: {request_id}")
        pages = (token_capacity + self.page_size - 1) // self.page_size
        if pages > len(self.free_pages):
            raise MemoryError("paged KV cache capacity exhausted")
        self.tables[request_id] = [self.free_pages.pop() for _ in range(pages)]
        self.lengths[request_id] = 0

    def append(self, request_id: str, keys: Tensor, values: Tensor) -> None:
        if request_id not in self.tables:
            raise KeyError(f"unknown request: {request_id}")
        if keys.shape != values.shape or keys.ndim != 4:
            raise ValueError("keys and values must match [layers, kv_heads, tokens, head_dim]")
        expected = self.storage.shape
        if keys.shape[0] != expected[1] or keys.shape[1] != expected[3] or keys.shape[3] != expected[5]:
            raise ValueError("keys and values do not match the configured cache dimensions")
        if keys.device != self.storage.device or keys.dtype != self.storage.dtype:
            raise ValueError("keys and values must match the cache device and dtype")
        start = self.lengths[request_id]
        count = keys.shape[2]
        if start + count > len(self.tables[request_id]) * self.page_size:
            raise MemoryError("request exceeds its reserved KV capacity")
        for offset in range(count):
            position = start + offset
            page = self.tables[request_id][position // self.page_size]
            slot = position % self.page_size
            self.storage[page, :, 0, :, slot].copy_(keys[:, :, offset])
            self.storage[page, :, 1, :, slot].copy_(values[:, :, offset])
        self.lengths[request_id] += count

    def materialize(self, request_id: str) -> tuple[Tensor, Tensor]:
        if request_id not in self.tables:
            raise KeyError(f"unknown request: {request_id}")
        length = self.lengths[request_id]
        pages = self.tables[request_id]
        chunks = [self.storage[page] for page in pages]
        combined = torch.cat(chunks, dim=3)[..., :length, :]
        return combined[:, 0], combined[:, 1]

    def release(self, request_id: str) -> None:
        if request_id not in self.tables:
            raise KeyError(f"unknown request: {request_id}")
        self.free_pages.extend(self.tables.pop(request_id))
        self.lengths.pop(request_id)


class PrefixCache:
    """Bounded LRU mapping from prompt token tuples to immutable cache objects."""

    def __init__(self, capacity: int = 32) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._values: OrderedDict[tuple[int, ...], object] = OrderedDict()

    def get(self, tokens: tuple[int, ...]) -> object | None:
        value = self._values.get(tokens)
        if value is not None:
            self._values.move_to_end(tokens)
        return value

    def put(self, tokens: tuple[int, ...], value: object) -> None:
        self._values[tokens] = value
        self._values.move_to_end(tokens)
        while len(self._values) > self.capacity:
            self._values.popitem(last=False)
