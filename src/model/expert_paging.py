"""Disk-backed MoE expert paging with a bounded LRU memory cache."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import torch


class ExpertPager:
    """Persist individual expert state dicts and page them into memory on demand."""

    def __init__(self, root: str | Path, *, max_resident: int = 2, map_location="cpu"):
        if max_resident < 1:
            raise ValueError("max_resident must be positive")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_resident = max_resident
        self.map_location = map_location
        self._cache = OrderedDict()

    def _path(self, index: int) -> Path:
        return self.root / f"expert-{index:06d}.pt"

    def store(self, index: int, state_dict: dict):
        torch.save(
            {"format": "gopi-expert-page-v1", "index": index, "state_dict": state_dict},
            self._path(index),
        )
        self._cache.pop(index, None)

    def load(self, index: int):
        if index in self._cache:
            value = self._cache.pop(index)
            self._cache[index] = value
            return value
        path = self._path(index)
        if not path.exists():
            raise FileNotFoundError(path)
        payload = torch.load(path, map_location=self.map_location, weights_only=False)
        if payload.get("format") != "gopi-expert-page-v1":
            raise ValueError("unsupported expert page")
        value = payload["state_dict"]
        self._cache[index] = value
        while len(self._cache) > self.max_resident:
            self._cache.popitem(last=False)
        return value

    def resident_indices(self):
        return tuple(self._cache.keys())

    def available_indices(self):
        return tuple(
            sorted(int(p.stem.split("-")[-1]) for p in self.root.glob("expert-*.pt"))
        )

    def apply(self, index: int, module, *, strict=True):
        state = self.load(index)
        result = module.load_state_dict(state, strict=strict)
        return result
