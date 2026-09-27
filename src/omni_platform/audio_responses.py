"""Audio-only Responses modality contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AudioResponse:
    audio: bytes
    mime_type: str
    transcript: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_response_payload(self) -> dict[str, Any]:
        return {
            "type": "audio",
            "mime_type": self.mime_type,
            "audio_bytes": self.audio,
            "transcript": self.transcript,
            "metadata": dict(self.metadata),
        }


def validate_audio_only_request(request: dict[str, Any]) -> None:
    if request.get("input") in (None, "") and request.get("audio") in (None, ""):
        raise ValueError(
            "audio-only response requires audio input or an audio generation request"
        )
    if request.get("text") not in (None, "") and request.get("audio_only"):
        raise ValueError("audio_only requests must not require text output")
