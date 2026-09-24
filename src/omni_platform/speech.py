"""Operational speech providers with lazy optional Hugging Face backends.

Capabilities are only advertised when dependencies and configured local/remote model ids
can be loaded successfully. Heavy models are never imported at module import time.
"""
from __future__ import annotations

import math
import os
import tempfile
import threading
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from .errors import DependencyMissingError, GenerationFailedError, MediaValidationError, ProviderUnavailableError
from .providers import GenerationArtifact, GenerationResult, ProviderContext


def _require_transformers():
    try:
        from transformers import pipeline  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency-dependent
        raise DependencyMissingError("speech provider requires transformers; install the 'speech' extra") from exc
    return pipeline


def _write_pcm16_wav(path: Path, samples: Any, sample_rate: int) -> Path:
    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover
        raise DependencyMissingError("speech output requires numpy") from exc
    arr = np.asarray(samples, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        raise GenerationFailedError("TTS model returned empty audio")
    peak = float(np.max(np.abs(arr)))
    if peak > 1.0:
        arr = arr / max(peak, 1e-8)
    pcm = (np.clip(arr, -1.0, 1.0) * 32767.0).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm.tobytes())
    return path


@dataclass(slots=True)
class SpeechConfig:
    model: str
    device: str | int = "cpu"
    language: str | None = None
    chunk_length_s: float | None = None


class _LazyPipeline:
    def __init__(self, task: str, config: SpeechConfig) -> None:
        self.task = task
        self.config = config
        self._pipe = None
        self._lock = threading.Lock()
        self._error: str | None = None

    def load(self):
        if self._pipe is not None:
            return self._pipe
        if self._error is not None:
            raise ProviderUnavailableError(self._error)
        with self._lock:
            if self._pipe is not None:
                return self._pipe
            try:
                pipeline = _require_transformers()
                kwargs: dict[str, Any] = {"task": self.task, "model": self.config.model}
                if self.config.device != "cpu":
                    kwargs["device"] = self.config.device
                self._pipe = pipeline(**kwargs)
            except Exception as exc:
                self._error = f"failed to load {self.task} model {self.config.model!r}: {exc}"
                raise ProviderUnavailableError(self._error) from exc
        return self._pipe

    def available(self) -> bool:
        if not self.config.model:
            return False
        try:
            self.load()
            return True
        except Exception:
            return False


class HuggingFaceASRProvider:
    id = "hf-asr"
    capabilities = frozenset({"speech_to_text", "audio_input"})

    def __init__(self, config: SpeechConfig) -> None:
        self.config = config
        self._runtime = _LazyPipeline("automatic-speech-recognition", config)

    @classmethod
    def from_env(cls) -> "HuggingFaceASRProvider | None":
        model = os.getenv("GOPI_ASR_MODEL", "").strip()
        if not model:
            return None
        return cls(SpeechConfig(model=model, device=os.getenv("GOPI_ASR_DEVICE", "cpu"), language=os.getenv("GOPI_ASR_LANGUAGE") or None))

    def is_available(self) -> bool:
        return self._runtime.available()

    def transcribe(self, request: dict[str, Any], context: ProviderContext) -> dict[str, Any]:
        path = Path(str(request.get("path", "")))
        if not path.is_file():
            raise MediaValidationError("audio file does not exist")
        pipe = self._runtime.load()
        kwargs: dict[str, Any] = {"return_timestamps": request.get("timestamps", False)}
        task = request.get("task")
        if task in {"translate", "transcribe"}:
            kwargs["generate_kwargs"] = {"task": task}
        language = request.get("language") or self.config.language
        if language:
            kwargs.setdefault("generate_kwargs", {})["language"] = language
        if self.config.chunk_length_s:
            kwargs["chunk_length_s"] = self.config.chunk_length_s
        if context.progress:
            context.progress(0.05, "transcribing")
        try:
            result = pipe(str(path), **kwargs)
        except Exception as exc:
            raise GenerationFailedError(f"speech transcription failed: {exc}") from exc
        if context.progress:
            context.progress(1.0, "completed")
        if isinstance(result, str):
            return {"text": result}
        if not isinstance(result, dict):
            return {"text": str(result)}
        return result


class HuggingFaceTTSProvider:
    id = "hf-tts"
    capabilities = frozenset({"text_to_speech", "audio_output"})

    def __init__(self, config: SpeechConfig, *, output_dir: str | Path = "outputs/tts") -> None:
        self.config = config
        self.output_dir = Path(output_dir)
        self._runtime = _LazyPipeline("text-to-speech", config)

    @classmethod
    def from_env(cls) -> "HuggingFaceTTSProvider | None":
        model = os.getenv("GOPI_TTS_MODEL", "").strip()
        if not model:
            return None
        return cls(SpeechConfig(model=model, device=os.getenv("GOPI_TTS_DEVICE", "cpu"), language=os.getenv("GOPI_TTS_LANGUAGE") or None), output_dir=os.getenv("GOPI_TTS_OUTPUT_DIR", "outputs/tts"))

    def is_available(self) -> bool:
        return self._runtime.available()

    def synthesize(self, request: dict[str, Any], context: ProviderContext) -> GenerationResult:
        text = str(request.get("text", "")).strip()
        if not text:
            raise MediaValidationError("text is required")
        if len(text) > int(os.getenv("GOPI_TTS_MAX_CHARS", "12000")):
            raise MediaValidationError("text exceeds TTS limit")
        pipe = self._runtime.load()
        if context.progress:
            context.progress(0.05, "synthesizing")
        try:
            result = pipe(text)
        except Exception as exc:
            raise GenerationFailedError(f"speech synthesis failed: {exc}") from exc
        if not isinstance(result, dict) or "audio" not in result:
            raise GenerationFailedError("TTS backend returned an unsupported result")
        sample_rate = int(result.get("sampling_rate") or result.get("sample_rate") or 16000)
        output = Path(request.get("output") or self.output_dir / f"{context.request_id}.wav")
        _write_pcm16_wav(output, result["audio"], sample_rate)
        if context.progress:
            context.progress(1.0, "completed")
        return GenerationResult(
            artifacts=[GenerationArtifact(output, "audio/wav", {"sample_rate": sample_rate})],
            usage={"audio_seconds": _wav_duration(output)},
            metadata={"provider": self.id, "model": self.config.model},
        )

    def stream(self, text: str, *, chunk_chars: int = 240, context: ProviderContext) -> Iterable[bytes]:
        """Chunked WAV synthesis for low-latency HTTP streaming.

        Each yielded item is a complete WAV chunk. Clients can play chunks sequentially.
        This intentionally avoids pretending every model supports native token/audio streaming.
        """
        text = text.strip()
        for index in range(0, len(text), max(32, chunk_chars)):
            part = text[index:index + max(32, chunk_chars)]
            with tempfile.TemporaryDirectory(prefix="gopi-tts-") as tmp:
                result = self.synthesize({"text": part, "output": str(Path(tmp) / "chunk.wav")}, context)
                yield result.artifacts[0].path.read_bytes()


def _wav_duration(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as wf:
            return wf.getnframes() / max(1, wf.getframerate())
    except Exception:
        return 0.0


class EnergyVAD:
    """Dependency-free PCM WAV VAD suitable for preprocessing and endpointing."""

    id = "energy-vad"
    capabilities = frozenset({"voice_activity_detection"})

    def is_available(self) -> bool:
        return True

    def detect(self, path: str | Path, *, frame_ms: int = 30, threshold_dbfs: float = -42.0, min_speech_ms: int = 180) -> list[dict[str, float]]:
        p = Path(path)
        try:
            import numpy as np
        except Exception as exc:  # pragma: no cover
            raise DependencyMissingError("VAD requires numpy") from exc
        with wave.open(str(p), "rb") as wf:
            if wf.getsampwidth() != 2:
                raise MediaValidationError("energy VAD currently requires 16-bit PCM WAV")
            channels, rate = wf.getnchannels(), wf.getframerate()
            raw = wf.readframes(wf.getnframes())
        samples = np.frombuffer(raw, dtype="<i2").astype(np.float32)
        if channels > 1:
            samples = samples.reshape(-1, channels).mean(axis=1)
        frame = max(1, int(rate * frame_ms / 1000))
        active: list[bool] = []
        for start in range(0, len(samples), frame):
            chunk = samples[start:start + frame]
            rms = float(np.sqrt(np.mean(np.square(chunk))) + 1e-8)
            dbfs = 20.0 * math.log10(rms / 32768.0 + 1e-12)
            active.append(dbfs >= threshold_dbfs)
        minimum = max(1, math.ceil(min_speech_ms / frame_ms))
        segments: list[dict[str, float]] = []
        start_idx: int | None = None
        for i, on in enumerate(active + [False]):
            if on and start_idx is None:
                start_idx = i
            elif not on and start_idx is not None:
                if i - start_idx >= minimum:
                    segments.append({"start": start_idx * frame_ms / 1000.0, "end": i * frame_ms / 1000.0})
                start_idx = None
        return segments


class SpeechToSpeechPipeline:
    id = "speech-to-speech-pipeline"
    capabilities = frozenset({"speech_to_speech", "audio_input", "audio_output"})

    def __init__(self, asr: HuggingFaceASRProvider, tts: HuggingFaceTTSProvider, transform: Callable[[str], str] | None = None) -> None:
        self.asr, self.tts, self.transform = asr, tts, transform

    def is_available(self) -> bool:
        return self.asr.is_available() and self.tts.is_available()

    def convert_speech(self, request: dict[str, Any], context: ProviderContext) -> GenerationResult:
        transcript = self.asr.transcribe({"path": request.get("path")}, context)
        text = str(transcript.get("text", ""))
        if self.transform is not None:
            text = self.transform(text)
        result = self.tts.synthesize({"text": text, "output": request.get("output")}, context)
        result.metadata.update({"transcript": transcript, "provider": self.id})
        return result
