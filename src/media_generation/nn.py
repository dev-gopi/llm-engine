"""Small reusable neural-network primitives for media diffusion models."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def timestep_embedding(timesteps: Tensor, dim: int, max_period: int = 10_000) -> Tensor:
    """Create sinusoidal timestep embeddings with shape ``[batch, dim]``."""
    if timesteps.ndim != 1:
        raise ValueError("timesteps must have shape [batch]")
    if dim < 2:
        raise ValueError("embedding dimension must be at least two")
    half = dim // 2
    frequencies = torch.exp(
        -math.log(max_period) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / half
    )
    values = timesteps.float()[:, None] * frequencies[None]
    result = torch.cat([torch.cos(values), torch.sin(values)], dim=-1)
    if dim % 2:
        result = torch.nn.functional.pad(result, (0, 1))
    return result


def valid_group_count(channels: int, preferred: int = 32) -> int:
    for groups in range(min(preferred, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class FiLM(nn.Module):
    """Feature-wise affine conditioning from one global embedding."""

    def __init__(self, embedding_size: int, channels: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(nn.SiLU(), nn.Linear(embedding_size, channels * 2))

    def forward(self, features: Tensor, embedding: Tensor) -> Tensor:
        scale, shift = self.projection(embedding).chunk(2, dim=-1)
        while scale.ndim < features.ndim:
            scale = scale.unsqueeze(-1)
            shift = shift.unsqueeze(-1)
        return features * (1 + scale) + shift
