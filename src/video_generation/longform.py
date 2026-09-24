"""Long-form video assembly helpers."""
from __future__ import annotations
import torch
from torch import Tensor


def blend_video_segments(segments: list[Tensor], overlap_frames: int) -> Tensor:
    """Blend ``[3,T,H,W]`` segments with linear temporal crossfades."""
    if not segments:
        raise ValueError("segments cannot be empty")
    if overlap_frames < 0:
        raise ValueError("overlap_frames cannot be negative")
    output = segments[0]
    for segment in segments[1:]:
        overlap = min(overlap_frames, output.shape[1], segment.shape[1])
        if overlap:
            fade = torch.linspace(0, 1, overlap, device=segment.device, dtype=segment.dtype)[None, :, None, None]
            blended = output[:, -overlap:] * (1 - fade) + segment[:, :overlap] * fade
            output = torch.cat([output[:, :-overlap], blended, segment[:, overlap:]], dim=1)
        else:
            output = torch.cat([output, segment], dim=1)
    return output
