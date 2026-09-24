"""Prepare mixed Responses API inputs for the existing native multimodal backend.

The adapter deliberately reuses the trained native vision runtime rather than
introducing a second vision stack. Audio is transcribed by a configured ASR
provider. Video is sampled safely with FFmpeg into image data URLs and may also
include an ASR transcript of its audio track.
"""
from __future__ import annotations

import asyncio
import base64
import os
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from media_generation.assets import AssetStore
from omni_platform.errors import MediaValidationError, ProviderUnavailableError
from omni_platform.ffmpeg import FFmpeg
from omni_platform.providers import ProviderContext
from omni_platform.speech import HuggingFaceASRProvider


@dataclass(slots=True)
class PreparedMultimodalInput:
    messages: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)


def _data_url(data: bytes, mime_type: str) -> str:
    return f"data:{mime_type};base64,{base64.b64encode(data).decode('ascii')}"


def _text_fallback(content: Any) -> str:
    if isinstance(content, str):
        return content
    values: list[str] = []
    for part in content or []:
        if isinstance(part, dict) and part.get("type") == "text":
            values.append(str(part.get("text", "")))
        elif isinstance(part, dict) and part.get("type") == "image_url":
            values.append("[image]")
    return "\n".join(v for v in values if v).strip()


def latest_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        value = _text_fallback(message.get("content"))
        if value:
            return value
    return ""


def _asset_record(store: AssetStore, asset_id: str, prefix: str):
    try:
        record = store.get(asset_id)
    except KeyError as exc:
        raise MediaValidationError(f"asset not found: {asset_id}") from exc
    if not record.mime_type.startswith(prefix):
        raise MediaValidationError(f"asset {asset_id} must be {prefix.rstrip('/')} media")
    return record


def _image_asset_url(store: AssetStore, asset_id: str) -> str:
    record = _asset_record(store, asset_id, "image/")
    path = Path(record.path)
    limit = int(os.getenv("GOPI_V7_MAX_INLINE_IMAGE_BYTES", str(16 * 1024 * 1024)))
    if record.size_bytes > limit:
        raise MediaValidationError(f"image asset exceeds inline vision limit of {limit} bytes")
    return _data_url(path.read_bytes(), record.mime_type)


def _probe_duration(probe: dict[str, Any]) -> float:
    try:
        return max(0.0, float((probe.get("format") or {}).get("duration") or 0.0))
    except (TypeError, ValueError):
        return 0.0


def _sample_times(duration: float, count: int) -> list[float]:
    count = max(1, count)
    if count == 1 or duration <= 0:
        return [0.0]
    margin = min(0.25, max(0.0, duration / 20.0))
    usable = max(0.0, duration - 2 * margin)
    return [margin + usable * i / (count - 1) for i in range(count)]


async def _transcribe_asset(
    store: AssetStore,
    asset_id: str,
    *,
    language: str | None,
    asr: HuggingFaceASRProvider | None,
    request_id: str,
) -> str:
    record = _asset_record(store, asset_id, "audio/")
    if asr is None or not asr.is_available():
        raise ProviderUnavailableError("audio input requires a configured and ready speech-to-text provider")
    result = await asyncio.to_thread(
        asr.transcribe,
        {"path": record.path, "language": language, "timestamps": False, "task": "transcribe"},
        ProviderContext(request_id=request_id),
    )
    text = str(result.get("text", "")).strip()
    if not text:
        raise MediaValidationError("audio transcription produced no text")
    return text


async def _video_parts(
    store: AssetStore,
    asset_id: str,
    *,
    max_frames: int | None,
    language: str | None,
    asr: HuggingFaceASRProvider | None,
    request_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    record = _asset_record(store, asset_id, "video/")
    ff = await asyncio.to_thread(FFmpeg)
    probe = await asyncio.to_thread(ff.probe, record.path)
    duration = _probe_duration(probe)
    hard_limit = max(1, min(32, int(os.getenv("GOPI_V7_VIDEO_MAX_FRAMES", "8"))))
    count = min(max_frames or int(os.getenv("GOPI_V7_VIDEO_DEFAULT_FRAMES", "6")), hard_limit)
    parts: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    transcript: str | None = None
    with tempfile.TemporaryDirectory(prefix="gopi-v7-video-") as tmp:
        root = Path(tmp)
        for index, at in enumerate(_sample_times(duration, count)):
            frame = root / f"frame-{index:03d}.jpg"
            await asyncio.to_thread(ff.extract_thumbnail, record.path, frame, at_seconds=at)
            parts.append({"type": "image_url", "image_url": {"url": _data_url(frame.read_bytes(), "image/jpeg"), "detail": "auto"}})
            events.append({"type": "response.input_video.frame.sampled", "index": index, "timestamp": at})
        if asr is not None and asr.is_available():
            audio = root / "audio.wav"
            try:
                await asyncio.to_thread(ff.extract_audio, record.path, audio)
                result = await asyncio.to_thread(
                    asr.transcribe,
                    {"path": str(audio), "language": language, "timestamps": False, "task": "transcribe"},
                    ProviderContext(request_id=request_id),
                )
                transcript = str(result.get("text", "")).strip() or None
            except Exception:
                # Video visual understanding remains useful when a file has no
                # audio stream or optional ASR fails. We report this in metadata
                # rather than silently claiming a transcript exists.
                transcript = None
    preface = f"[Video input: {count} sampled frames"
    if duration:
        preface += f", duration {duration:.2f}s"
    preface += "]"
    if transcript:
        preface += f"\n[Audio transcript]\n{transcript}"
        events.append({"type": "response.input_video.transcription.completed", "text": transcript})
    return ([{"type": "text", "text": preface}] + parts), {
        "asset_id": asset_id,
        "duration_seconds": duration,
        "sampled_frames": count,
        "audio_transcribed": bool(transcript),
    }, events


async def prepare_responses_input(
    input_value: str | list[Any],
    *,
    asset_store: AssetStore | None = None,
    asr: HuggingFaceASRProvider | None = None,
) -> PreparedMultimodalInput:
    if isinstance(input_value, str):
        return PreparedMultimodalInput(messages=[{"role": "user", "content": input_value}])
    store = asset_store or AssetStore(os.getenv("GOPI_MEDIA_ASSET_DIR", "outputs/api_media/assets"))
    messages: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {"audio_inputs": [], "video_inputs": [], "image_inputs": 0}
    events: list[dict[str, Any]] = []
    request_id = f"prep_{uuid.uuid4().hex}"
    for item in input_value:
        raw = item.model_dump(mode="json") if hasattr(item, "model_dump") else dict(item)
        role = raw.get("role", "user")
        content = raw.get("content")
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue
        out: list[dict[str, Any]] = []
        for part in content or []:
            if hasattr(part, "model_dump"):
                part = part.model_dump(mode="json")
            kind = part.get("type")
            if kind == "input_text":
                out.append({"type": "text", "text": part["text"]})
            elif kind == "input_image":
                url = part.get("image_url") or _image_asset_url(store, part["asset_id"])
                out.append({"type": "image_url", "image_url": {"url": url, "detail": part.get("detail", "auto")}})
                metadata["image_inputs"] += 1
            elif kind == "input_audio":
                transcript = await _transcribe_asset(
                    store, part["asset_id"], language=part.get("language"), asr=asr, request_id=request_id
                )
                out.append({"type": "text", "text": f"[Audio transcript]\n{transcript}"})
                metadata["audio_inputs"].append({"asset_id": part["asset_id"], "transcribed": True})
                events.append({"type": "response.input_audio.transcription.completed", "asset_id": part["asset_id"], "text": transcript})
            elif kind == "input_video":
                video_parts, video_meta, video_events = await _video_parts(
                    store,
                    part["asset_id"],
                    max_frames=part.get("max_frames"),
                    language=part.get("language"),
                    asr=asr,
                    request_id=request_id,
                )
                out.extend(video_parts)
                metadata["video_inputs"].append(video_meta)
                events.extend(video_events)
            else:
                raise MediaValidationError(f"unsupported Responses input part: {kind}")
        messages.append({"role": role, "content": out})
    return PreparedMultimodalInput(messages=messages, metadata=metadata, events=events)


async def synthesize_response_audio(
    text: str,
    *,
    tts: Any,
    asset_store: AssetStore | None = None,
    request_id: str,
) -> dict[str, Any]:
    """Synthesize assistant text and persist it as a protected media asset.

    The Responses API returns an asset identifier rather than an absolute server
    path or an unbounded base64 payload.
    """
    if tts is None or not tts.is_available():
        raise ProviderUnavailableError("audio output requires a configured and ready text-to-speech provider")
    store = asset_store or AssetStore(os.getenv("GOPI_MEDIA_ASSET_DIR", "outputs/api_media/assets"))
    result = await asyncio.to_thread(
        tts.synthesize,
        {"text": text},
        ProviderContext(request_id=request_id),
    )
    if not result.artifacts:
        raise MediaValidationError("text-to-speech provider returned no audio artifact")
    artifact = result.artifacts[0]
    payload = artifact.path.read_bytes()
    record = store.put(payload, mime_type=artifact.mime_type or "audio/wav", filename=f"{request_id}.wav")
    return {
        "asset_id": record.id,
        "mime_type": record.mime_type,
        "size_bytes": record.size_bytes,
        "sha256": record.sha256,
        "metadata": dict(artifact.metadata),
    }
