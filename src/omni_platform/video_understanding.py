"""Video understanding pipeline: safe frame sampling + optional ASR + multimodal summarizer."""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Callable

from .errors import MediaValidationError, ProviderUnavailableError
from .ffmpeg import FFmpeg
from .providers import ProviderContext


class VideoUnderstandingPipeline:
    id = "video-understanding-pipeline"
    capabilities = frozenset({"video_understanding", "video_input"})

    def __init__(
        self,
        *,
        frame_describer: Callable[[Path, str | None], str] | None = None,
        summarizer: Callable[[str], str] | None = None,
        asr_provider: Any | None = None,
        ffmpeg: FFmpeg | None = None,
        max_frames: int = 12,
    ) -> None:
        self.frame_describer = frame_describer
        self.summarizer = summarizer
        self.asr_provider = asr_provider
        self.ffmpeg = ffmpeg
        self.max_frames = max(1, max_frames)

    def is_available(self) -> bool:
        return self.frame_describer is not None and self.summarizer is not None

    def understand_video(self, request: dict[str, Any], context: ProviderContext) -> dict[str, Any]:
        if not self.is_available():
            raise ProviderUnavailableError("video understanding requires a frame describer and summarizer")
        path = Path(str(request.get("path", "")))
        if not path.is_file():
            raise MediaValidationError("video file does not exist")
        ff = self.ffmpeg or FFmpeg()
        probe = ff.probe(path)
        duration = _duration(probe)
        frame_count = min(int(request.get("max_frames", self.max_frames)), self.max_frames)
        question = request.get("question")
        descriptions: list[str] = []
        transcript: dict[str, Any] | None = None
        with tempfile.TemporaryDirectory(prefix="gopi-video-understand-") as tmp:
            root = Path(tmp)
            for index, at in enumerate(_sample_times(duration, frame_count)):
                if context.cancellation_token is not None and getattr(context.cancellation_token, "cancelled", False):
                    raise RuntimeError("video understanding cancelled")
                image = ff.extract_thumbnail(path, root / f"frame-{index:03d}.jpg", at_seconds=at)
                descriptions.append(self.frame_describer(image, question))
                if context.progress:
                    context.progress(0.7 * ((index + 1) / frame_count), "analyzing_frames")
            if self.asr_provider is not None and self.asr_provider.is_available():
                audio = root / "audio.wav"
                try:
                    ff.extract_audio(path, audio)
                    transcript = self.asr_provider.transcribe({"path": str(audio)}, context)
                except Exception:
                    transcript = None
        evidence = "\n".join(f"Frame {i + 1}: {text}" for i, text in enumerate(descriptions))
        if transcript and transcript.get("text"):
            evidence += f"\nTranscript: {transcript['text']}"
        prompt = (f"Question: {question}\n" if question else "") + "Video evidence:\n" + evidence
        summary = self.summarizer(prompt)
        if context.progress:
            context.progress(1.0, "completed")
        return {"summary": summary, "frames": descriptions, "transcript": transcript, "duration_seconds": duration}


def _duration(probe: dict[str, Any]) -> float:
    try:
        return max(0.0, float((probe.get("format") or {}).get("duration") or 0.0))
    except Exception:
        return 0.0


def _sample_times(duration: float, count: int) -> list[float]:
    if count <= 1 or duration <= 0:
        return [0.0]
    margin = min(0.25, duration / 10.0)
    span = max(0.0, duration - margin * 2)
    return [margin + span * i / (count - 1) for i in range(count)]
