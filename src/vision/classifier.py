"""Image-classification head for supervised VisionEncoder training."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import Tensor, nn

from .encoder import VisionEncoder


class VisionClassifier(nn.Module):
    def __init__(self, encoder: VisionEncoder, num_classes: int, dropout: float = 0.0) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least two")
        self.encoder = encoder
        self.num_classes = num_classes
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(encoder.hidden_size, num_classes))

    def forward(self, images: Tensor) -> Tensor:
        return self.head(self.encoder.pooled(images))

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "VisionClassifier":
        return cls(
            VisionEncoder.from_config(config),
            num_classes=int(config["num_classes"]),
            dropout=float(config.get("classifier_dropout", 0.0)),
        )
