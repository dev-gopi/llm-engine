"""Unified operational and native multimodal Omni endpoints."""
from __future__ import annotations

import base64
import json
import os
import secrets
import time
import uuid
from typing import Any

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from media_generation.assets import AssetStore
from api.responses import ResponseInput, ResponseInputText, ResponseInputVideo
from omni_platform.capabilities import build_capability_snapshot
from omni_platform.errors import OmniError
from omni_platform.ffmpeg import FFmpeg
from omni_platform.multimodal_input import latest_text, prepare_responses_input
from omni_platform.observability import METRICS
from omni_platform.providers import ProviderContext, ProviderRegistry
from omni_platform.speech import EnergyVAD, HuggingFaceASRProvider, HuggingFaceTTSProvider, SpeechToSpeechPipeline
from serving.runtime import ServingError
from serving.schemas import GenerateRequest


class TranscriptionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    asset_id: str
    language: str | None = Field(default=None, max_length=32)
    timestamps: bool = False


class SpeechRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input: str = Field(min_length=1, max_length=12000)
    voice: str | None = Field(default=None, max_length=128)
    language: str | None = Field(default=None, max_length=32)
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    response_format: str = Field(default="wav", pattern=r"^wav$")


class VADRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    asset_id: str
    threshold_dbfs: float = Field(default=-42.0, ge=-90, le=-1)
    min_speech_ms: int = Field(default=180, ge=30, le=5000)


class SpeechToSpeechRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    asset_id: str
    language: str | None = None


class VideoUnderstandRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    asset_id: str = Field(pattern=r"^asset_[0-9a-f]{32}$")
    prompt: str = Field(default="Describe and summarize this video accurately.", min_length=1, max_length=8192)
    language: str | None = Field(default=None, max_length=32)
    max_frames: int = Field(default=6, ge=1, le=32)
    max_output_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=0.2, ge=0, le=2)
    seed: int | None = Field(default=42, ge=0, le=2**63 - 1)


def _asset_store() -> AssetStore:
    return AssetStore(os.getenv("GOPI_MEDIA_ASSET_DIR", "outputs/api_media/assets"))


def _require_auth(authorization: str | None) -> None:
    key = os.getenv("GOPI_API_KEY", "")
    if not key:
        if os.getenv("GOPI_ALLOW_UNAUTHENTICATED_MEDIA", "0") == "1":
            return
        raise HTTPException(503, "GOPI_API_KEY is required for Omni media endpoints")
    supplied = (authorization or "").removeprefix("Bearer ")
    if not supplied or not secrets.compare_digest(supplied, key):
        raise HTTPException(401, "invalid bearer token")


def _providers() -> tuple[ProviderRegistry, HuggingFaceASRProvider | None, HuggingFaceTTSProvider | None]:
    registry = ProviderRegistry()
    asr = HuggingFaceASRProvider.from_env()
    tts = HuggingFaceTTSProvider.from_env()
    for provider in (asr, tts):
        if provider is not None:
            registry.register(provider)
    if asr is not None and tts is not None:
        registry.register(SpeechToSpeechPipeline(asr, tts))
    registry.register(EnergyVAD())
    return registry, asr, tts


def _native_backend(runtime: Any) -> Any:
    backend = getattr(runtime, "backend", runtime)
    return getattr(backend, "backend", backend)


def create_omni_speech_router() -> APIRouter:
    router = APIRouter(prefix="/v1", tags=["omni-speech"])

    @router.get("/platform/capabilities/verify")
    async def verify_capabilities(authorization: str | None = Header(default=None)):
        _require_auth(authorization)
        registry, _, _ = _providers()
        started = time.perf_counter()
        providers = registry.describe()
        caps = registry.capability_set()
        snapshot = build_capability_snapshot(provider_capabilities=caps)
        METRICS.set("capability_verification_seconds", time.perf_counter() - started)
        return {"object": "platform.capability.verification", **snapshot.as_dict(), "providers": providers}

    @router.get("/metrics/prometheus", response_class=PlainTextResponse)
    async def prometheus_metrics():
        return METRICS.prometheus()

    @router.post("/audio/transcriptions")
    async def audio_transcriptions(req: TranscriptionRequest, authorization: str | None = Header(default=None)):
        _require_auth(authorization)
        _, asr, _ = _providers()
        if asr is None or not asr.is_available():
            raise HTTPException(503, "speech-to-text provider is not configured and ready")
        path = _asset_store().resolve(req.asset_id)
        request_id = f"asr_{uuid.uuid4().hex}"
        started = time.perf_counter()
        try:
            result = asr.transcribe({"path": str(path), "language": req.language, "timestamps": req.timestamps, "task": "transcribe"}, ProviderContext(request_id=request_id))
        finally:
            METRICS.set("asr_last_duration_seconds", time.perf_counter() - started)
        METRICS.inc("asr_requests_total")
        return {"id": request_id, "object": "audio.transcription", **result}

    @router.post("/audio/translations")
    async def audio_translations(req: TranscriptionRequest, authorization: str | None = Header(default=None)):
        _require_auth(authorization)
        _, asr, _ = _providers()
        if asr is None or not asr.is_available():
            raise HTTPException(503, "audio translation provider is not configured and ready")
        if os.getenv("GOPI_ASR_SUPPORTS_TRANSLATION", "0") != "1":
            raise HTTPException(501, "configured ASR model does not advertise translation support")
        path = _asset_store().resolve(req.asset_id)
        result = asr.transcribe({"path": str(path), "language": req.language, "timestamps": req.timestamps, "task": "translate"}, ProviderContext(request_id=f"atr_{uuid.uuid4().hex}"))
        METRICS.inc("audio_translation_requests_total")
        return {"object": "audio.translation", **result}

    @router.post("/audio/speech")
    async def audio_speech(req: SpeechRequest, authorization: str | None = Header(default=None)):
        _require_auth(authorization)
        _, _, tts = _providers()
        if tts is None or not tts.is_available():
            raise HTTPException(503, "text-to-speech provider is not configured and ready")
        request_id = f"tts_{uuid.uuid4().hex}"
        result = tts.synthesize({"text": req.input}, ProviderContext(request_id=request_id))
        METRICS.inc("tts_requests_total")
        return FileResponse(result.artifacts[0].path, media_type="audio/wav", filename=f"{request_id}.wav")

    @router.post("/audio/speech/stream")
    async def audio_speech_stream(req: SpeechRequest, authorization: str | None = Header(default=None)):
        _require_auth(authorization)
        _, _, tts = _providers()
        if tts is None or not tts.is_available():
            raise HTTPException(503, "text-to-speech provider is not configured and ready")
        context = ProviderContext(request_id=f"ttsstream_{uuid.uuid4().hex}")
        def events():
            for index, chunk in enumerate(tts.stream(req.input, context=context)):
                payload = {"type": "response.audio.delta", "index": index, "audio": base64.b64encode(chunk).decode("ascii"), "format": "wav"}
                yield f"data: {json.dumps(payload)}\n\n"
            yield 'data: {"type":"response.audio.done"}\n\n'
        return StreamingResponse(events(), media_type="text/event-stream")

    @router.post("/audio/voice-activity")
    async def audio_vad(req: VADRequest, authorization: str | None = Header(default=None)):
        _require_auth(authorization)
        path = _asset_store().resolve(req.asset_id)
        segments = EnergyVAD().detect(path, threshold_dbfs=req.threshold_dbfs, min_speech_ms=req.min_speech_ms)
        return {"object": "audio.voice_activity", "segments": segments}

    @router.post("/audio/speech-to-speech")
    async def speech_to_speech(req: SpeechToSpeechRequest, authorization: str | None = Header(default=None)):
        _require_auth(authorization)
        _, asr, tts = _providers()
        if asr is None or tts is None or not asr.is_available() or not tts.is_available():
            raise HTTPException(503, "speech-to-speech requires ready ASR and TTS providers")
        path = _asset_store().resolve(req.asset_id)
        result = SpeechToSpeechPipeline(asr, tts).convert_speech({"path": str(path)}, ProviderContext(request_id=f"s2s_{uuid.uuid4().hex}"))
        METRICS.inc("speech_to_speech_requests_total")
        return FileResponse(result.artifacts[0].path, media_type="audio/wav")

    return router


def create_omni_video_router(runtime: Any) -> APIRouter:
    """Create native multimodal video-understanding routes."""
    router = APIRouter(prefix="/v1", tags=["omni-video"])

    @router.get("/platform/capabilities/runtime")
    async def runtime_capabilities(authorization: str | None = Header(default=None)):
        _require_auth(authorization)
        native = _native_backend(runtime)
        vision = bool(getattr(native, "supports_vision", False) and getattr(native, "multimodal_generator", None) is not None)
        try:
            FFmpeg()
            ffmpeg_ready = True
        except OmniError:
            ffmpeg_ready = False
        asr = HuggingFaceASRProvider.from_env()
        asr_ready = bool(asr is not None and asr.is_available())
        video_ready = vision and ffmpeg_ready
        return {
            "object": "platform.runtime.capabilities",
            "capabilities": {"vision": vision, "video_input": video_ready, "video_understanding": video_ready, "speech_to_text": asr_ready, "audio_input": asr_ready},
            "evidence": {
                "vision": ["native_multimodal_checkpoint"] if vision else [],
                "video_understanding": ["native_vision+safe_ffmpeg_sampling"] if video_ready else [],
                "speech_to_text": ["ready_asr_provider"] if asr_ready else [],
            },
        }

    @router.post("/videos/understand")
    async def understand_video(req: VideoUnderstandRequest, authorization: str | None = Header(default=None)):
        _require_auth(authorization)
        native = _native_backend(runtime)
        if not bool(getattr(native, "supports_vision", False) and getattr(native, "multimodal_generator", None) is not None):
            raise HTTPException(503, "video understanding requires a ready native multimodal vision checkpoint")
        item = ResponseInput(role="user", content=[
            ResponseInputText(text=req.prompt),
            ResponseInputVideo(asset_id=req.asset_id, max_frames=req.max_frames, language=req.language),
        ])
        try:
            prepared = await prepare_responses_input([item], asr=HuggingFaceASRProvider.from_env())
        except OmniError as exc:
            raise HTTPException(exc.http_status, detail={"code": exc.code, "message": exc.message}) from exc
        generation = GenerateRequest(prompt=latest_text(prepared.messages), max_tokens=req.max_output_tokens,
                                     temperature=req.temperature, seed=req.seed, mode="precise")
        generation._chat_messages = prepared.messages
        try:
            result = await runtime.generate(generation)
        except ServingError:
            # Let the application-level ServingError handler preserve the
            # established 503/504/429 status mapping for runtime failures.
            raise
        except Exception as exc:
            raise HTTPException(500, "video understanding failed") from exc
        return {
            "id": f"vunder_{uuid.uuid4().hex}", "object": "video.understanding",
            "model": getattr(native, "model_name", None) or os.getenv("GOPI_MODEL_NAME", "gopi"),
            "text": result.text, "input_metadata": prepared.metadata,
            "usage": {"prompt_tokens": result.prompt_tokens, "completion_tokens": result.completion_tokens,
                      "total_tokens": result.prompt_tokens + result.completion_tokens},
        }

    return router
