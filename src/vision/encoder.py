"""Compact Vision Transformer suitable for training on limited hardware."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .patch_embedding import PatchEmbedding


class VisionBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        heads: int,
        ffn_hidden_size: int,
        *,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_size % heads:
            raise ValueError("hidden_size must be divisible by heads")
        self.attention_norm = nn.LayerNorm(hidden_size)
        self.attention = nn.MultiheadAttention(
            hidden_size, heads, dropout=dropout, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden_size, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        normalized = self.attention_norm(hidden_states)
        attended, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        hidden_states = hidden_states + attended
        return hidden_states + self.ffn(self.ffn_norm(hidden_states))


class VisionEncoder(nn.Module):
    """Encode a fixed-size RGB image as patch tokens and a pooled class token."""

    def __init__(
        self,
        image_size: int = 128,
        patch_size: int = 16,
        channels: int = 3,
        hidden_size: int = 192,
        layers: int = 4,
        heads: int = 3,
        ffn_hidden_size: int = 768,
        dropout: float = 0.05,
        initializer_range: float = 0.02,
        strict_image_size: bool = True,
        pool_type: str = "cls",
    ) -> None:
        super().__init__()
        if layers <= 0 or ffn_hidden_size <= 0:
            raise ValueError("layers and ffn_hidden_size must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1")
        self.hidden_size = hidden_size
        if pool_type not in {"cls", "mean"}:
            raise ValueError("pool_type must be cls or mean")
        self.patch_embedding = PatchEmbedding(
            image_size, patch_size, channels, hidden_size,
            strict_image_size=strict_image_size,
        )
        self.pool_type = pool_type
        token_count = self.patch_embedding.num_patches + 1
        self.class_token = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.position_embedding = nn.Parameter(torch.empty(1, token_count, hidden_size))
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            VisionBlock(hidden_size, heads, ffn_hidden_size, dropout=dropout)
            for _ in range(layers)
        )
        self.norm = nn.LayerNorm(hidden_size)
        nn.init.normal_(self.class_token, std=initializer_range)
        nn.init.normal_(self.position_embedding, std=initializer_range)

    def forward(self, images: Tensor) -> Tensor:
        patches = self.patch_embedding(images)
        class_token = self.class_token.expand(images.shape[0], -1, -1)
        hidden_states = torch.cat((class_token, patches), dim=1)
        hidden_states = self.dropout(hidden_states + self._position_embedding(images))
        for block in self.blocks:
            hidden_states = block(hidden_states)
        return self.norm(hidden_states)

    def pooled(self, images: Tensor) -> Tensor:
        encoded = self(images)
        return encoded[:, 0] if self.pool_type == "cls" else encoded[:, 1:].mean(dim=1)

    def _position_embedding(self, images: Tensor) -> Tensor:
        height = images.shape[-2] // self.patch_embedding.patch_size
        width = images.shape[-1] // self.patch_embedding.patch_size
        if height == self.patch_embedding.grid_size and width == self.patch_embedding.grid_size:
            return self.position_embedding
        cls_position = self.position_embedding[:, :1]
        patch_position = self.position_embedding[:, 1:].reshape(
            1, self.patch_embedding.grid_size, self.patch_embedding.grid_size, self.hidden_size
        ).permute(0, 3, 1, 2)
        patch_position = F.interpolate(
            patch_position.float(), size=(height, width), mode="bicubic", align_corners=False
        ).to(self.position_embedding.dtype)
        return torch.cat((cls_position, patch_position.flatten(2).transpose(1, 2)), dim=1)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "VisionEncoder":
        return cls(
            image_size=int(config.get("image_size", 128)),
            patch_size=int(config.get("patch_size", 16)),
            channels=int(config.get("channels", 3)),
            hidden_size=int(config.get("hidden_size", 192)),
            layers=int(config.get("layers", 4)),
            heads=int(config.get("heads", 3)),
            ffn_hidden_size=int(config.get("ffn_hidden_size", 768)),
            dropout=float(config.get("dropout", 0.05)),
            initializer_range=float(config.get("initializer_range", 0.02)),
            strict_image_size=bool(config.get("strict_image_size", True)),
            pool_type=str(config.get("pool_type", "cls")),
        )
