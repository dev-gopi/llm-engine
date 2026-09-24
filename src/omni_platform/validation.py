"""Centralized media request validation and safe limits."""
from __future__ import annotations

from dataclasses import dataclass

from .errors import GenerationValidationError


@dataclass(frozen=True, slots=True)
class GenerationLimits:
    max_prompt_chars: int = 8192
    max_outputs: int = 8
    max_image_pixels: int = 4096 * 4096
    max_audio_seconds: float = 1800.0
    max_video_seconds: float = 120.0
    max_video_fps: int = 60
    max_video_frames: int = 3600


def validate_prompt(prompt: str, limits: GenerationLimits) -> str:
    prompt = prompt.strip()
    if not prompt:
        raise GenerationValidationError("prompt cannot be empty")
    if len(prompt) > limits.max_prompt_chars:
        raise GenerationValidationError("prompt exceeds configured length limit")
    return prompt


def validate_image_size(width: int, height: int, limits: GenerationLimits) -> None:
    if width < 16 or height < 16 or width * height > limits.max_image_pixels:
        raise GenerationValidationError("requested image dimensions exceed configured limits")


def validate_video(*, frames: int, fps: int, limits: GenerationLimits) -> None:
    if frames < 1 or frames > limits.max_video_frames:
        raise GenerationValidationError("requested frame count exceeds configured limits")
    if fps < 1 or fps > limits.max_video_fps:
        raise GenerationValidationError("requested FPS exceeds configured limits")
    if frames / fps > limits.max_video_seconds:
        raise GenerationValidationError("requested video duration exceeds configured limits")
