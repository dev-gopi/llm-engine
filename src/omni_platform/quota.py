"""Modality-aware resource accounting primitives."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ResourceUsage:
    text_tokens: int = 0
    images: int = 0
    audio_seconds: float = 0.0
    video_seconds: float = 0.0
    gpu_seconds: float = 0.0
    storage_bytes: int = 0

    def weighted_units(self, *, audio_weight: float = 5.0, video_weight: float = 50.0, image_weight: float = 100.0, gpu_weight: float = 10.0) -> float:
        return (
            self.text_tokens
            + self.images * image_weight
            + self.audio_seconds * audio_weight
            + self.video_seconds * video_weight
            + self.gpu_seconds * gpu_weight
        )
