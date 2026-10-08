"""Audio-only Responses modality contracts."""

from __future__ import annotations

import io
import wave
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AudioStreamChunk:
    chunk_index: int
    audio: bytes
    is_final: bool = False
    format: str = "wav"
    sample_rate: int = 24000

    def to_payload(self) -> dict[str, Any]:
        return {
            "type": "audio_chunk",
            "index": self.chunk_index,
            "audio_bytes": self.audio,
            "format": self.format,
            "sample_rate": self.sample_rate,
            "is_final": self.is_final,
        }


def encode_pcm_to_wav(
    pcm_bytes: bytes,
    *,
    channels: int = 1,
    sample_width: int = 2,
    frame_rate: int = 24000,
) -> bytes:
    """Encode raw PCM audio samples into a standard RIFF/WAV byte container."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(sample_width)
        wav.setframerate(frame_rate)
        wav.writeframes(pcm_bytes)
    return buffer.getvalue()


def chunk_audio_stream(
    audio: bytes,
    *,
    chunk_size_bytes: int = 4096,
    format: str = "wav",
    sample_rate: int = 24000,
) -> Iterator[AudioStreamChunk]:
    """Yield sequential streaming chunks for audio streaming endpoints."""
    if not audio:
        yield AudioStreamChunk(
            chunk_index=0,
            audio=b"",
            is_final=True,
            format=format,
            sample_rate=sample_rate,
        )
        return

    total_chunks = max(1, (len(audio) + chunk_size_bytes - 1) // chunk_size_bytes)
    for index in range(total_chunks):
        start = index * chunk_size_bytes
        end = min(len(audio), start + chunk_size_bytes)
        chunk_data = audio[start:end]
        is_final = index == total_chunks - 1
        yield AudioStreamChunk(
            chunk_index=index,
            audio=chunk_data,
            is_final=is_final,
            format=format,
            sample_rate=sample_rate,
        )


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


__all__ = [
    "AudioStreamChunk",
    "AudioResponse",
    "encode_pcm_to_wav",
    "chunk_audio_stream",
    "validate_audio_only_request",
]
