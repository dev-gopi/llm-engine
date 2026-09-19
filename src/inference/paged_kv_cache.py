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
        quantization: str = "none",
    ) -> None:
        if min(num_pages, page_size, layers, kv_heads, head_dim) < 1:
            raise ValueError("paged cache dimensions must be positive")
        if quantization not in {"none", "int8"}:
            raise ValueError("quantization must be none or int8")
        if quantization == "int8" and not dtype.is_floating_point:
            raise ValueError("INT8 KV quantization requires a floating-point output dtype")
        shape = (num_pages, layers, 2, kv_heads, page_size, head_dim)
        self.quantization = quantization
        self.dtype = dtype
        self.storage = torch.empty(shape, device=device, dtype=torch.int8 if quantization == "int8" else dtype)
        self.scales = (
            torch.ones((*shape[:-1], 1), device=device, dtype=torch.float32)
            if quantization == "int8" else None
        )
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
        if keys.device != self.storage.device or keys.dtype != self.dtype:
            raise ValueError("keys and values must match the cache device and configured dtype")
        start = self.lengths[request_id]
        count = keys.shape[2]
        if start + count > len(self.tables[request_id]) * self.page_size:
            raise MemoryError("request exceeds its reserved KV capacity")
        for offset in range(count):
            position = start + offset
            page = self.tables[request_id][position // self.page_size]
            slot = position % self.page_size
            if self.quantization == "int8":
                self._store_int8(page, 0, slot, keys[:, :, offset])
                self._store_int8(page, 1, slot, values[:, :, offset])
            else:
                self.storage[page, :, 0, :, slot].copy_(keys[:, :, offset])
                self.storage[page, :, 1, :, slot].copy_(values[:, :, offset])
        self.lengths[request_id] += count

    def materialize(self, request_id: str) -> tuple[Tensor, Tensor]:
        if request_id not in self.tables:
            raise KeyError(f"unknown request: {request_id}")
        length = self.lengths[request_id]
        pages = self.tables[request_id]
        chunks = [self._read_page(page) for page in pages]
        combined = torch.cat(chunks, dim=3)[..., :length, :]
        return combined[:, 0], combined[:, 1]

    @property
    def storage_nbytes(self) -> int:
        """Allocated KV payload plus scales, for explicit precision trade-offs."""
        total = self.storage.numel() * self.storage.element_size()
        return total + (0 if self.scales is None else self.scales.numel() * self.scales.element_size())

    def _store_int8(self, page: int, kind: int, slot: int, values: Tensor) -> None:
        assert self.scales is not None
        scale = values.detach().abs().amax(dim=-1, keepdim=True).to(torch.float32) / 127
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        self.storage[page, :, kind, :, slot].copy_(torch.clamp(torch.round(values / scale.to(values.dtype)), -127, 127).to(torch.int8))
        self.scales[page, :, kind, :, slot].copy_(scale)

    def _read_page(self, page: int) -> Tensor:
        stored = self.storage[page]
        if self.scales is None:
            return stored
        return (stored.to(torch.float32) * self.scales[page]).to(self.dtype)

    def release(self, request_id: str) -> None:
        if request_id not in self.tables:
            raise KeyError(f"unknown request: {request_id}")
        self.free_pages.extend(self.tables.pop(request_id))
        self.lengths.pop(request_id)

    def layer_cache(self, request_ids: list[str], layer: int) -> "PagedLayerKVCache":
        """Expose page tables for one transformer layer without materializing KV."""
        if not request_ids or any(request_id not in self.tables for request_id in request_ids):
            raise KeyError("all paged-cache request IDs must be reserved")
        if not 0 <= layer < self.storage.shape[1]:
            raise ValueError("layer is outside the configured paged cache")
        return PagedLayerKVCache(self, tuple(request_ids), layer)


class PagedLayerKVCache:
    """Read-only page-table view consumed directly by decode attention.

    It intentionally exposes page-sized tensors rather than a contiguous KV
    tensor.  ``pending`` holds only the newly projected decode token so the
    serving runtime can append it to its owning request after a batched call.
    """

    is_paged_kv_cache = True

    def __init__(self, allocator: PagedKVCache, request_ids: tuple[str, ...], layer: int) -> None:
        self.allocator = allocator
        self.request_ids = request_ids
        self.layer = layer
        self.pending: tuple[Tensor, Tensor] | None = None

    @property
    def length(self) -> int:
        return max(self.allocator.lengths[request_id] for request_id in self.request_ids)

    def pages(self) -> list[tuple[Tensor, Tensor, Tensor]]:
        """Return ``(key, value, valid)`` page tensors for every table slot.

        Key/value tensors are ``[batch, kv_heads, page, head_dim]`` and the
        boolean validity mask is ``[batch, page]``.  No request cache is
        concatenated or materialized.
        """
        tables = [self.allocator.tables[request_id] for request_id in self.request_ids]
        widths = max(map(len, tables))
        lengths = torch.tensor(
            [self.allocator.lengths[request_id] for request_id in self.request_ids],
            device=self.allocator.storage.device,
        )
        pages: list[tuple[Tensor, Tensor, Tensor]] = []
        for slot in range(widths):
            identifiers = torch.tensor(
                [table[slot] if slot < len(table) else 0 for table in tables],
                device=self.allocator.storage.device,
            )
            stored = self.allocator.storage[identifiers, self.layer]
            if self.allocator.scales is not None:
                stored = (stored.to(torch.float32) * self.allocator.scales[identifiers, self.layer]).to(self.allocator.dtype)
            valid = (
                torch.arange(self.allocator.page_size, device=stored.device)
                .unsqueeze(0)
                < (lengths - slot * self.allocator.page_size).unsqueeze(1)
            )
            pages.append((stored[:, 0], stored[:, 1], valid))
        return pages

    def record_pending(self, key: Tensor, value: Tensor) -> None:
        if key.shape != value.shape or key.ndim != 4 or key.shape[0] != len(self.request_ids):
            raise ValueError("pending paged KV must match the active batch")
        self.pending = (key.detach(), value.detach())


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


class PagedPrefixCache:
    """LRU prefix cache backed by the fixed-page KV allocator for batch-one generation."""

    def __init__(self, allocator: PagedKVCache, *, capacity: int = 32) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.allocator = allocator
        self.capacity = capacity
        self.entries: OrderedDict[tuple[int, ...], tuple[str, Tensor]] = OrderedDict()
        self.counter = 0

    def get(self, tokens: tuple[int, ...]) -> tuple[Tensor, tuple[tuple[Tensor, Tensor], ...]] | None:
        entry = self.entries.get(tokens)
        if entry is None:
            return None
        self.entries.move_to_end(tokens)
        request_id, logits = entry
        keys, values = self.allocator.materialize(request_id)
        cache = tuple(
            (keys[layer].unsqueeze(0), values[layer].unsqueeze(0))
            for layer in range(keys.shape[0])
        )
        return logits, cache

    def put(self, tokens: tuple[int, ...], logits: Tensor, cache) -> None:
        if tokens in self.entries:
            request_id, _ = self.entries.pop(tokens)
            self.allocator.release(request_id)
        required_pages = (len(tokens) + self.allocator.page_size - 1) // self.allocator.page_size
        if required_pages > len(self.allocator.free_pages) + sum(
            len(self.allocator.tables[request_id]) for request_id, _ in self.entries.values()
        ):
            raise MemoryError("prefix requires more pages than the cache capacity")
        while len(self.entries) >= self.capacity or required_pages > len(self.allocator.free_pages):
            _, (expired, _) = self.entries.popitem(last=False)
            self.allocator.release(expired)
        self.counter += 1
        request_id = f"prefix-{self.counter}"
        self.allocator.reserve(request_id, len(tokens))
        keys = torch.stack([layer[0].squeeze(0) for layer in cache])
        values = torch.stack([layer[1].squeeze(0) for layer in cache])
        self.allocator.append(request_id, keys, values)
        self.entries[tokens] = (request_id, logits.detach().clone())
