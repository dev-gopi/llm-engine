"""Long-form audio orchestration using overlapping diffusion windows."""
from __future__ import annotations
import math
import torch
from torch import Tensor


def crossfade_chunks(chunks: list[Tensor], overlap_samples: int) -> Tensor:
    if not chunks:
        raise ValueError("chunks cannot be empty")
    if overlap_samples < 0:
        raise ValueError("overlap_samples cannot be negative")
    output = chunks[0].flatten()
    for chunk in chunks[1:]:
        chunk = chunk.flatten()
        overlap = min(overlap_samples, output.numel(), chunk.numel())
        if overlap:
            fade_in = torch.linspace(0, 1, overlap, device=chunk.device, dtype=chunk.dtype)
            fade_out = 1 - fade_in
            blended = output[-overlap:] * fade_out + chunk[:overlap] * fade_in
            output = torch.cat([output[:-overlap], blended, chunk[overlap:]])
        else:
            output = torch.cat([output, chunk])
    return output


def plan_windows(total_samples: int, window_samples: int, overlap_samples: int) -> list[int]:
    if min(total_samples, window_samples) <= 0:
        raise ValueError("sample counts must be positive")
    if not 0 <= overlap_samples < window_samples:
        raise ValueError("overlap must satisfy 0 <= overlap < window")
    if total_samples <= window_samples:
        return [window_samples]
    stride = window_samples - overlap_samples
    count = math.ceil((total_samples - window_samples) / stride) + 1
    return [window_samples] * count
