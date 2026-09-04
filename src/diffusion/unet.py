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
    def __init__(self, in_channels: int, out_channels: int, condition_size: int,
                 dropout: float = 0.0) -> None:
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
        self.dropout = nn.Dropout2d(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, inputs: Tensor, condition: Tensor) -> Tensor:
        hidden = self.conv1(F.silu(self.norm1(inputs)))
        hidden = hidden + self.condition(condition).unsqueeze(-1).unsqueeze(-1)
        hidden = self.conv2(self.dropout(F.silu(self.norm2(hidden))))
        return hidden + self.skip(inputs)


class SpatialAttention(nn.Module):
    def __init__(self, channels: int, heads: int = 4) -> None:
        super().__init__()
        if channels % heads:
            raise ValueError("attention channels must be divisible by heads")
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.attention = nn.MultiheadAttention(channels, heads, batch_first=True)

    def forward(self, inputs: Tensor) -> Tensor:
        batch, channels, height, width = inputs.shape
        tokens = self.norm(inputs).flatten(2).transpose(1, 2)
        attended, _ = self.attention(tokens, tokens, tokens, need_weights=False)
        return inputs + attended.transpose(1, 2).reshape(batch, channels, height, width)


class SpatialCrossAttention(nn.Module):
    def __init__(self, channels: int, condition_size: int, heads: int = 4) -> None:
        super().__init__()
        if channels % heads:
            raise ValueError("cross-attention channels must be divisible by heads")
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.context_projection = nn.Linear(condition_size, channels)
        self.attention = nn.MultiheadAttention(channels, heads, batch_first=True)

    def forward(self, inputs: Tensor, context: Tensor,
                context_mask: Tensor | None = None) -> Tensor:
        if context.ndim != 3 or context.shape[0] != inputs.shape[0]:
            raise ValueError("text_context must have shape [batch, sequence, condition_size]")
        batch, channels, height, width = inputs.shape
        query = self.norm(inputs).flatten(2).transpose(1, 2)
        context = self.context_projection(context.to(query.dtype))
        key_padding_mask = None if context_mask is None else ~context_mask.bool()
        attended, _ = self.attention(
            query, context, context, key_padding_mask=key_padding_mask, need_weights=False
        )
        return inputs + attended.transpose(1, 2).reshape(batch, channels, height, width)


class SmallUNet(nn.Module):
    """Predict Gaussian noise, optionally conditioned on a fixed-size text vector."""

    def __init__(
        self,
        image_channels: int = 3,
        base_channels: int = 64,
        condition_size: int = 256,
        dropout: float = 0.0,
        attention_heads: int = 4,
        use_attention: bool = False,
        num_classes: int | None = None,
        use_cross_attention: bool = False,
    ) -> None:
        super().__init__()
        if min(image_channels, base_channels, condition_size) <= 0:
            raise ValueError("channel and condition sizes must be positive")
        self.image_channels = image_channels
        self.condition_size = condition_size
        if num_classes is not None and num_classes < 2:
            raise ValueError("num_classes must be at least two when provided")
        self.num_classes = num_classes
        self.null_class_id = num_classes if num_classes is not None else None
        self.class_embedding = (
            nn.Embedding(num_classes + 1, condition_size) if num_classes is not None else None
        )
        self.time_embedding = TimeEmbedding(condition_size)
        self.text_projection = nn.Linear(condition_size, condition_size)
        self.input = nn.Conv2d(image_channels, base_channels, 3, padding=1)
        if not 0 <= dropout < 1:
            raise ValueError("dropout must satisfy 0 <= dropout < 1")
        self.down1 = ResidualBlock(base_channels, base_channels, condition_size, dropout)
        self.downsample1 = nn.Conv2d(base_channels, base_channels * 2, 4, stride=2, padding=1)
        self.down2 = ResidualBlock(base_channels * 2, base_channels * 2, condition_size, dropout)
        self.downsample2 = nn.Conv2d(base_channels * 2, base_channels * 4, 4, stride=2, padding=1)
        self.middle = ResidualBlock(base_channels * 4, base_channels * 4, condition_size, dropout)
        self.middle_attention = SpatialAttention(base_channels * 4, attention_heads) if use_attention else nn.Identity()
        self.middle_cross_attention = (
            SpatialCrossAttention(base_channels * 4, condition_size, attention_heads)
            if use_cross_attention else None
        )
        self.upsample2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, stride=2, padding=1)
        self.up2 = ResidualBlock(base_channels * 4, base_channels * 2, condition_size, dropout)
        self.upsample1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1)
        self.up1 = ResidualBlock(base_channels * 2, base_channels, condition_size, dropout)
        self.output_norm = nn.GroupNorm(min(8, base_channels), base_channels)
        self.output = nn.Conv2d(base_channels, image_channels, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self, noisy_images: Tensor, timesteps: Tensor, text_condition: Tensor | None = None,
        class_labels: Tensor | None = None,
        text_context: Tensor | None = None,
        text_context_mask: Tensor | None = None,
    ) -> Tensor:
        if noisy_images.ndim != 4 or noisy_images.shape[1] != self.image_channels:
            raise ValueError("noisy_images has an incompatible shape")
        if noisy_images.shape[-1] % 4 or noisy_images.shape[-2] % 4:
            raise ValueError("image height and width must be divisible by four")
        if timesteps.shape != (noisy_images.shape[0],):
            raise ValueError("timesteps must have shape [batch]")
        condition = self.time_embedding(timesteps).to(noisy_images.dtype)
        if sum(value is not None for value in (text_condition, class_labels, text_context)) > 1:
            raise ValueError("provide only one conditioning input")
        if class_labels is not None:
            if self.class_embedding is None:
                raise ValueError("class_labels require num_classes in the model configuration")
            if class_labels.shape != (noisy_images.shape[0],):
                raise ValueError("class_labels must have shape [batch]")
            if class_labels.dtype not in (torch.int32, torch.int64):
                raise TypeError("class_labels must use an integer dtype")
            condition = condition + self.class_embedding(class_labels).to(condition.dtype)
        if text_condition is not None:
            if text_condition.shape != (noisy_images.shape[0], self.condition_size):
                raise ValueError("text_condition has an incompatible shape")
            condition = condition + self.text_projection(text_condition.to(condition.dtype))
        first = self.down1(self.input(noisy_images), condition)
        second = self.down2(self.downsample1(first), condition)
        middle = self.middle(self.downsample2(second), condition)
        middle = self.middle_attention(middle)
        if text_context is not None:
            if self.middle_cross_attention is None:
                raise ValueError("text_context requires use_cross_attention in model configuration")
            middle = self.middle_cross_attention(middle, text_context, text_context_mask)
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
            dropout=float(config.get("dropout", 0.0)),
            attention_heads=int(config.get("attention_heads", 4)),
            use_attention=bool(config.get("use_attention", False)),
            num_classes=(int(config["num_classes"]) if config.get("num_classes") is not None else None),
            use_cross_attention=bool(config.get("use_cross_attention", False)),
        )
