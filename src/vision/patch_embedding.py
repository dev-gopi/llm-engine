"""Turn images into a sequence of non-overlapping patch embeddings."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class PatchEmbedding(nn.Module):
    def __init__(self, image_size: int, patch_size: int, channels: int, hidden_size: int,
                 *, strict_image_size: bool = True) -> None:
        super().__init__()
        if image_size <= 0 or patch_size <= 0 or image_size % patch_size:
            raise ValueError("image_size must be positive and divisible by patch_size")
        if channels <= 0 or hidden_size <= 0:
            raise ValueError("channels and hidden_size must be positive")
        self.image_size = image_size
        self.patch_size = patch_size
        self.channels = channels
        self.hidden_size = hidden_size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size**2
        self.strict_image_size = strict_image_size
        self.projection = nn.Conv2d(
            channels, hidden_size, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 4:
            raise ValueError("images must have shape [batch, channels, height, width]")
        if images.shape[1] != self.channels:
            raise ValueError(f"images must have {self.channels} channels")
        height, width = images.shape[-2:]
        if self.strict_image_size and (height, width) != (self.image_size, self.image_size):
            expected = (self.channels, self.image_size, self.image_size)
            raise ValueError(f"images must have trailing shape {expected}")
        if height % self.patch_size or width % self.patch_size:
            raise ValueError("image height and width must be divisible by patch_size")
        if not images.is_floating_point():
            raise TypeError("images must use a floating-point dtype")
        patches = self.projection(images)
        return patches.flatten(2).transpose(1, 2)
