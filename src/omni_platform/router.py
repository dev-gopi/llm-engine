"""Deterministic multimodal routing based on explicit modality and requested operation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .errors import UnsupportedCapabilityError

Modality = Literal["text", "image", "audio", "video"]


@dataclass(frozen=True, slots=True)
class RouteDecision:
    capability: str
    input_modalities: tuple[Modality, ...]
    output_modality: Modality


_CAPABILITY_MAP: dict[tuple[tuple[Modality, ...], Modality, str], str] = {
    (("text",), "text", "generate"): "chat",
    (("image",), "text", "understand"): "vision",
    (("image", "text"), "text", "understand"): "vision",
    (("text",), "image", "generate"): "image_generation",
    (("image",), "image", "edit"): "image_editing",
    (("text",), "audio", "speech"): "text_to_speech",
    (("audio",), "text", "transcribe"): "speech_to_text",
    (("audio",), "audio", "convert"): "speech_to_speech",
    (("text",), "audio", "generate"): "audio_generation",
    (("text",), "video", "generate"): "video_generation",
    (("image", "text"), "video", "generate"): "image_to_video",
    (("video",), "text", "understand"): "video_understanding",
}


def route_multimodal(
    *,
    input_modalities: list[str] | tuple[str, ...],
    output_modality: str,
    operation: str,
) -> RouteDecision:
    normalized_input = tuple(sorted({str(v).lower() for v in input_modalities}))
    key = (normalized_input, str(output_modality).lower(), str(operation).lower())
    capability = _CAPABILITY_MAP.get(key)  # type: ignore[arg-type]
    if capability is None:
        raise UnsupportedCapabilityError(
            f"no route for inputs={normalized_input}, output={output_modality}, operation={operation}"
        )
    return RouteDecision(capability, normalized_input, str(output_modality).lower())  # type: ignore[arg-type]
