"""Sequence packing and dynamic-padding efficiency calculations."""
from __future__ import annotations

import math
from collections.abc import Iterable


def packing_efficiency(lengths: Iterable[int], capacity: int) -> dict[str, float | int]:
    lengths = [int(value) for value in lengths]
    if capacity < 2 or any(value < 1 or value > capacity for value in lengths):
        raise ValueError("sequence lengths must be within 1..capacity")
    if not lengths:
        return {"sequences": 0, "useful_tokens": 0, "allocated_tokens": 0, "wasted_tokens": 0, "utilization": 0.0}
    allocated = len(lengths) * capacity
    useful = sum(lengths)
    wasted = allocated - useful
    return {
        "sequences": len(lengths), "useful_tokens": useful, "allocated_tokens": allocated,
        "wasted_tokens": wasted, "utilization": round(useful / allocated, 8),
    }


def batch_padding_efficiency(lengths: Iterable[int], batch_size: int, pad_to_multiple_of: int | None = None) -> dict[str, float | int]:
    lengths = [int(value) for value in lengths]
    if batch_size < 1 or any(value < 1 for value in lengths):
        raise ValueError("batch_size must be positive and lengths must be positive")
    useful = sum(lengths)
    allocated = 0
    for start in range(0, len(lengths), batch_size):
        batch = lengths[start:start + batch_size]
        width = max(batch)
        if pad_to_multiple_of:
            if pad_to_multiple_of < 1:
                raise ValueError("pad_to_multiple_of must be positive")
            width = math.ceil(width / pad_to_multiple_of) * pad_to_multiple_of
        allocated += width * len(batch)
    return {
        "sequences": len(lengths), "useful_tokens": useful, "allocated_tokens": allocated,
        "padding_tokens": allocated - useful,
        "utilization": round(useful / allocated, 8) if allocated else 0.0,
    }
