"""Compact conditional U-Net noise predictor."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class TimeEmbedding(nn.Module):
    def __init__(self, size: int) -> None:
        super().__init__()
        if size <= 0 or size % 2:
            raise ValueError("time embedding size must be a positive even integer")
        self.size = size
        self.network = nn.Sequential(nn.Linear(size, size * 4), nn.SiLU(), nn.Linear(size * 4, size))

    def forward(self, timesteps: Tensor) -> Tensor:
        half = self.size // 2
        frequencies = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=timesteps.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat((angles.sin(), angles.cos()), dim=1)
        return self.network(embedding.to(self.network[0].weight.dtype))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, condition_size: int) -> None:
        super().__init__()
        groups = min(8, in_channels)
        while in_channels % groups:
            groups -= 1
        out_groups = min(8, out_channels)
        while out_channels % out_groups:
            out_groups -= 1
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.condition = nn.Linear(condition_size, out_channels)
        self.norm2 = nn.GroupNorm(out_groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, inputs: Tensor, condition: Tensor) -> Tensor:
        hidden = self.conv1(F.silu(self.norm1(inputs)))
        hidden = hidden + self.condition(condition).unsqueeze(-1).unsqueeze(-1)
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        return hidden + self.skip(inputs)


class SmallUNet(nn.Module):
    """Predict Gaussian noise, optionally conditioned on a fixed-size text vector."""

    def __init__(
        self,
        image_channels: int = 3,
        base_channels: int = 64,
        condition_size: int = 256,
    ) -> None:
        super().__init__()
        if min(image_channels, base_channels, condition_size) <= 0:
            raise ValueError("channel and condition sizes must be positive")
        self.image_channels = image_channels
        self.condition_size = condition_size
        self.time_embedding = TimeEmbedding(condition_size)
        self.text_projection = nn.Linear(condition_size, condition_size)
        self.input = nn.Conv2d(image_channels, base_channels, 3, padding=1)
        self.down1 = ResidualBlock(base_channels, base_channels, condition_size)
        self.downsample1 = nn.Conv2d(base_channels, base_channels * 2, 4, stride=2, padding=1)
        self.down2 = ResidualBlock(base_channels * 2, base_channels * 2, condition_size)
        self.downsample2 = nn.Conv2d(base_channels * 2, base_channels * 4, 4, stride=2, padding=1)
        self.middle = ResidualBlock(base_channels * 4, base_channels * 4, condition_size)
        self.upsample2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, stride=2, padding=1)
        self.up2 = ResidualBlock(base_channels * 4, base_channels * 2, condition_size)
        self.upsample1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1)
        self.up1 = ResidualBlock(base_channels * 2, base_channels, condition_size)
        self.output_norm = nn.GroupNorm(min(8, base_channels), base_channels)
        self.output = nn.Conv2d(base_channels, image_channels, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self, noisy_images: Tensor, timesteps: Tensor, text_condition: Tensor | None = None
    ) -> Tensor:
        if noisy_images.ndim != 4 or noisy_images.shape[1] != self.image_channels:
            raise ValueError("noisy_images has an incompatible shape")
        if noisy_images.shape[-1] % 4 or noisy_images.shape[-2] % 4:
            raise ValueError("image height and width must be divisible by four")
        if timesteps.shape != (noisy_images.shape[0],):
            raise ValueError("timesteps must have shape [batch]")
        condition = self.time_embedding(timesteps).to(noisy_images.dtype)
        if text_condition is not None:
            if text_condition.shape != (noisy_images.shape[0], self.condition_size):
                raise ValueError("text_condition has an incompatible shape")
            condition = condition + self.text_projection(text_condition.to(condition.dtype))
        first = self.down1(self.input(noisy_images), condition)
        second = self.down2(self.downsample1(first), condition)
        middle = self.middle(self.downsample2(second), condition)
        hidden = self.upsample2(middle)
        hidden = self.up2(torch.cat((hidden, second), dim=1), condition)
        hidden = self.upsample1(hidden)
        hidden = self.up1(torch.cat((hidden, first), dim=1), condition)
        return self.output(F.silu(self.output_norm(hidden)))

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "SmallUNet":
        return cls(
            image_channels=int(config.get("image_channels", 3)),
            base_channels=int(config.get("base_channels", 64)),
            condition_size=int(config.get("condition_size", 256)),
        )
