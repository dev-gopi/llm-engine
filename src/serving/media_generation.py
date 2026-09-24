"""Production audio/video generation routes with sync and asynchronous job APIs."""
from __future__ import annotations

import asyncio
import gc
import json
import os
import secrets
import threading
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch
from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool

from media_generation.assets import AssetStore
from media_generation.jobs import MediaJob, MediaJobStore
from media_generation.progress import CancellationToken
from media_generation.registry import MediaModelSpec, load_registry
from media_generation.runtime import AudioGenerator, VideoGenerator
from media_generation.storyboard import build_storyboard
from omni_platform.capabilities import build_capability_snapshot
from omni_platform.resources import gpu_report
from omni_platform.router import route_multimodal


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AudioGenerationRequest(_Strict):
    model: str | None = None
    prompt: str = Field(min_length=1, max_length=4096)
    negative_prompt: str = Field(default="", max_length=4096)
    seconds: float | None = Field(default=None, gt=0, le=300)
    guidance_scale: float | None = Field(default=None, ge=0, le=20)
    steps: int | None = Field(default=None, ge=1, le=250)
    seed: int | None = Field(default=None, ge=0, le=2**63 - 1)
    preset: str | None = Field(default=None, pattern="^(draft|balanced|quality|max_quality)$")
    style: str | None = None
    long_form: bool = False
    overlap_seconds: float = Field(default=0.5, ge=0, le=10)
    init_audio_asset_id: str | None = None
    strength: float = Field(default=1.0, ge=0, le=1)
    preserve_start_seconds: float | None = Field(default=None, ge=0, le=300)
    preserve_end_seconds: float | None = Field(default=None, ge=0, le=300)


class VideoGenerationRequest(_Strict):
    model: str | None = None
    prompt: str = Field(min_length=1, max_length=4096)
    negative_prompt: str = Field(default="", max_length=4096)
    frames: int | None = Field(default=None, ge=1, le=240)
    height: int | None = Field(default=None, ge=32, le=1024)
    width: int | None = Field(default=None, ge=32, le=1024)
    fps: int | None = Field(default=None, ge=1, le=60)
    guidance_scale: float | None = Field(default=None, ge=0, le=20)
    steps: int | None = Field(default=None, ge=1, le=250)
    seed: int | None = Field(default=None, ge=0, le=2**63 - 1)
    crf: int = Field(default=18, ge=0, le=51)
    preset: str | None = Field(default=None, pattern="^(draft|balanced|quality|max_quality)$")
    style: str | None = None
    segments: int = Field(default=1, ge=1, le=16)
    overlap_frames: int = Field(default=2, ge=0, le=32)
    init_image_asset_id: str | None = None
    init_video_asset_id: str | None = None
    strength: float = Field(default=1.0, ge=0, le=1)
    scene_prompts: list[str] | None = Field(default=None, max_length=16)
    camera: str | None = Field(
        default=None,
        pattern="^(static|dolly_in|dolly_out|pan_left|pan_right|orbit|handheld|drone)$",
    )


class MediaGenerationRequest(_Strict):
    kind: str = Field(pattern="^(audio|video)$")
    model: str | None = None
    prompt: str = Field(min_length=1, max_length=4096)
    negative_prompt: str = Field(default="", max_length=4096)
    preset: str | None = Field(default=None, pattern="^(draft|balanced|quality|max_quality)$")
    style: str | None = None
    seed: int | None = Field(default=None, ge=0, le=2**63 - 1)
    steps: int | None = Field(default=None, ge=1, le=250)
    guidance_scale: float | None = Field(default=None, ge=0, le=20)
    # Audio controls.
    seconds: float | None = Field(default=None, gt=0, le=300)
    long_form: bool = False
    overlap_seconds: float = Field(default=0.5, ge=0, le=10)
    init_audio_asset_id: str | None = None
    preserve_start_seconds: float | None = Field(default=None, ge=0, le=300)
    preserve_end_seconds: float | None = Field(default=None, ge=0, le=300)
    # Video controls.
    frames: int | None = Field(default=None, ge=1, le=240)
    height: int | None = Field(default=None, ge=32, le=1024)
    width: int | None = Field(default=None, ge=32, le=1024)
    fps: int | None = Field(default=None, ge=1, le=60)
    crf: int = Field(default=18, ge=0, le=51)
    segments: int = Field(default=1, ge=1, le=16)
    overlap_frames: int = Field(default=2, ge=0, le=32)
    init_image_asset_id: str | None = None
    init_video_asset_id: str | None = None
    strength: float = Field(default=1.0, ge=0, le=1)
    scene_prompts: list[str] | None = Field(default=None, max_length=16)
    camera: str | None = Field(
        default=None,
        pattern="^(static|dolly_in|dolly_out|pan_left|pan_right|orbit|handheld|drone)$",
    )


def _require_auth(auth: str | None) -> None:
    expected = os.getenv("GOPI_API_KEY")
    if not expected:
        return
    supplied = auth[7:] if auth and auth.startswith("Bearer ") else ""
    if not secrets.compare_digest(supplied, expected):
        raise HTTPException(status_code=401, detail="invalid API key")


class _BusyError(RuntimeError):
    pass


class _Gate:
    def __init__(self) -> None:
        self.sem = threading.BoundedSemaphore(max(1, int(os.getenv("GOPI_MEDIA_MAX_CONCURRENCY", "1"))))
        self.timeout = max(0.0, float(os.getenv("GOPI_MEDIA_QUEUE_TIMEOUT_SECONDS", "2")))

    def __enter__(self):
        if not self.sem.acquire(timeout=self.timeout):
            raise _BusyError("media generation capacity is busy; retry later")
        return self

    def __exit__(self, *_):
        self.sem.release()


_gate = _Gate()
_cache_lock = threading.Lock()
_cache: OrderedDict[tuple[str, str], object] = OrderedDict()
_active_lock = threading.Lock()
_active_tokens: dict[str, CancellationToken] = {}
_job_workers = max(1, int(os.getenv("GOPI_MEDIA_JOB_WORKERS", os.getenv("GOPI_MEDIA_MAX_CONCURRENCY", "1"))))
_job_queue_limit = max(0, int(os.getenv("GOPI_MEDIA_MAX_QUEUED_JOBS", "16")))
_executor = ThreadPoolExecutor(max_workers=_job_workers, thread_name_prefix="gopi-media")
_job_slots = threading.BoundedSemaphore(_job_workers + _job_queue_limit)
_job_store_instance: MediaJobStore | None = None
_asset_store_instance: AssetStore | None = None


def _job_store() -> MediaJobStore:
    global _job_store_instance
    if _job_store_instance is None:
        _job_store_instance = MediaJobStore(os.getenv("GOPI_MEDIA_JOB_DB", "outputs/api_media/media_jobs.sqlite3"))
    return _job_store_instance


def _asset_store() -> AssetStore:
    global _asset_store_instance
    if _asset_store_instance is None:
        _asset_store_instance = AssetStore(os.getenv("GOPI_MEDIA_ASSET_DIR", "outputs/api_media/assets"))
    return _asset_store_instance


def _specs() -> dict[str, MediaModelSpec]:
    path = os.getenv("GOPI_MEDIA_MODEL_REGISTRY")
    result: dict[str, MediaModelSpec] = {}
    if path:
        result.update(load_registry(path))
    if os.getenv("GOPI_AUDIO_CONFIG") and os.getenv("GOPI_AUDIO_CHECKPOINT"):
        result.setdefault(
            "default-audio",
            MediaModelSpec(
                "default-audio", "audio", Path(os.environ["GOPI_AUDIO_CONFIG"]), Path(os.environ["GOPI_AUDIO_CHECKPOINT"])
            ),
        )
    if os.getenv("GOPI_VIDEO_CONFIG") and os.getenv("GOPI_VIDEO_CHECKPOINT"):
        result.setdefault(
            "default-video",
            MediaModelSpec(
                "default-video", "video", Path(os.environ["GOPI_VIDEO_CONFIG"]), Path(os.environ["GOPI_VIDEO_CHECKPOINT"])
            ),
        )
    return result


def _runtime(kind: str, model_id: str | None):
    specs = _specs()
    candidates = [s for s in specs.values() if s.kind == kind]
    if model_id:
        spec = specs.get(model_id)
    else:
        spec = next((s for s in candidates if s.id == f"default-{kind}"), candidates[0] if candidates else None)
    if spec is None or spec.kind != kind:
        raise RuntimeError(f"no configured {kind} model" if not model_id else f"unknown {kind} model: {model_id}")
    key = (kind, spec.id)
    with _cache_lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]
        runtime = AudioGenerator(spec.config, spec.checkpoint) if kind == "audio" else VideoGenerator(spec.config, spec.checkpoint)
        _cache[key] = runtime
        limit = max(1, int(os.getenv("GOPI_MEDIA_RUNTIME_CACHE_SIZE", "2")))
        while len(_cache) > limit:
            _, evicted = _cache.popitem(last=False)
            del evicted
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return runtime


def _asset_path(asset_id: str | None) -> Path | None:
    if not asset_id:
        return None
    try:
        return _asset_store().resolve(asset_id)
    except KeyError as error:
        raise ValueError(f"unknown media asset: {asset_id}") from error


def _video_scene_prompts(req: VideoGenerationRequest | MediaGenerationRequest) -> list[str] | None:
    if not req.scene_prompts and not req.camera:
        return None
    shots = build_storyboard(req.prompt, req.scene_prompts, camera=req.camera)
    if req.scene_prompts:
        return [shot.prompt for shot in shots]
    # A camera-only request applies to every segment.
    return [shots[0].prompt for _ in range(req.segments)]


def _run_audio(req: AudioGenerationRequest | MediaGenerationRequest, out: Path, *, progress_callback=None, cancellation_token=None):
    return _runtime("audio", req.model).generate(
        prompt=req.prompt,
        negative_prompt=req.negative_prompt,
        output=out,
        seconds=req.seconds,
        seed=req.seed,
        preset=req.preset,
        style=req.style,
        steps=req.steps,
        guidance_scale=req.guidance_scale,
        long_form=req.long_form,
        overlap_seconds=req.overlap_seconds,
        init_audio=_asset_path(req.init_audio_asset_id),
        strength=req.strength,
        preserve_start_seconds=req.preserve_start_seconds,
        preserve_end_seconds=req.preserve_end_seconds,
        progress_callback=progress_callback,
        cancellation_token=cancellation_token,
        metadata=False,
    )


def _run_video(req: VideoGenerationRequest | MediaGenerationRequest, out: Path, *, progress_callback=None, cancellation_token=None):
    scene_prompts = _video_scene_prompts(req)
    if scene_prompts and len(scene_prompts) != req.segments:
        raise ValueError("scene_prompts length must equal segments")
    return _runtime("video", req.model).generate(
        prompt=req.prompt,
        negative_prompt=req.negative_prompt,
        output=out,
        frames=req.frames,
        height=req.height,
        width=req.width,
        fps=req.fps,
        seed=req.seed,
        preset=req.preset,
        style=req.style,
        steps=req.steps,
        guidance_scale=req.guidance_scale,
        crf=req.crf,
        segments=req.segments,
        overlap_frames=req.overlap_frames,
        init_image=_asset_path(req.init_image_asset_id),
        init_video=_asset_path(req.init_video_asset_id),
        strength=req.strength,
        scene_prompts=scene_prompts,
        progress_callback=progress_callback,
        cancellation_token=cancellation_token,
        metadata=False,
    )


def _response(path: Path):
    keep = os.getenv("GOPI_MEDIA_KEEP_OUTPUTS", "0").lower() in {"1", "true", "yes"}
    background = None if keep else BackgroundTask(path.unlink, missing_ok=True)
    return FileResponse(
        path,
        media_type="audio/wav" if path.suffix == ".wav" else "video/mp4",
        filename=path.name,
        background=background,
    )


def _job_dict(job: MediaJob) -> dict[str, Any]:
    payload = {
        "id": job.id,
        "object": "media.generation.job",
        "kind": job.kind,
        "status": job.status,
        "progress": round(job.progress, 6),
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "error": job.error,
        "cancel_requested": job.cancel_requested,
        "result": job.result,
    }
    if job.status == "succeeded":
        payload["content_url"] = f"/v1/media/jobs/{job.id}/content"
    return payload


def _execute_job(job_id: str) -> None:
    store = _job_store()
    token = CancellationToken.create()
    try:
        job = store.get(job_id)
    except KeyError:
        _job_slots.release()
        return
    with _active_lock:
        _active_tokens[job_id] = token
    try:
        if job.cancel_requested:
            store.update(job_id, status="cancelled", progress=0.0)
            return
        store.update(job_id, status="running", progress=0.001)
        request = MediaGenerationRequest(**job.request)
        output_dir = Path(os.getenv("GOPI_MEDIA_OUTPUT_DIR", "outputs/api_media/jobs"))
        output_dir.mkdir(parents=True, exist_ok=True)
        suffix = ".wav" if request.kind == "audio" else ".mp4"
        out = output_dir / f"{job_id}{suffix}"

        def progress(done: int, total: int) -> None:
            if store.is_cancel_requested(job_id):
                token.cancel()
            fraction = 0.0 if total <= 0 else done / total
            store.update(job_id, progress=min(0.99, max(0.001, fraction)))

        with _gate:
            if request.kind == "audio":
                path, details = _run_audio(request, out, progress_callback=progress, cancellation_token=token)
            else:
                path, details = _run_video(request, out, progress_callback=progress, cancellation_token=token)
        if token.cancelled or store.is_cancel_requested(job_id):
            Path(path).unlink(missing_ok=True)
            store.update(job_id, status="cancelled", progress=0.0)
            return
        result = {
            "path": str(path),
            "media_type": "audio/wav" if request.kind == "audio" else "video/mp4",
            "filename": Path(path).name,
            "quality": details.get("quality"),
            "settings": details.get("settings"),
        }
        store.update(job_id, status="succeeded", progress=1.0, result=result)
    except Exception as error:
        status = "cancelled" if token.cancelled or store.is_cancel_requested(job_id) else "failed"
        store.update(job_id, status=status, error=str(error), progress=0.0 if status == "cancelled" else None)
    finally:
        with _active_lock:
            _active_tokens.pop(job_id, None)
        _job_slots.release()


def _submit_job(request: MediaGenerationRequest, *, idempotency_key: str | None = None) -> MediaJob:
    if idempotency_key:
        existing = _job_store().get_by_idempotency(idempotency_key.strip())
        if existing is not None:
            return existing
    if not _job_slots.acquire(blocking=False):
        raise _BusyError("media job queue is full; retry later")
    try:
        job, created = _job_store().create_or_get(
            request.kind, request.model_dump(exclude_none=True), idempotency_key=idempotency_key
        )
        if not created:
            _job_slots.release()
            return job
        _executor.submit(_execute_job, job.id)
        return job
    except BaseException:
        _job_slots.release()
        raise


def create_media_router() -> APIRouter:
    router = APIRouter(prefix="/v1", tags=["media-generation"])

    @router.get("/media/models")
    async def media_models():
        return {
            "object": "list",
            "data": [
                {
                    "id": s.id, "kind": s.kind, "description": s.description,
                    "revision": s.revision, "capabilities": list(s.capabilities),
                    "metadata": s.metadata, "object": "media.model",
                }
                for s in _specs().values()
            ],
        }

    @router.get("/media/models/{model_id}")
    async def media_model(model_id: str):
        spec = _specs().get(model_id)
        if spec is None:
            raise HTTPException(404, "media model not found")
        return {
            "id": spec.id, "kind": spec.kind, "description": spec.description,
            "revision": spec.revision, "capabilities": list(spec.capabilities),
            "metadata": spec.metadata, "object": "media.model",
        }

    @router.get("/media/capabilities")
    async def capabilities():
        specs = _specs()
        advertised = {cap for spec in specs.values() for cap in spec.capabilities}
        # Conservative inference from actually configured model kinds. Advanced capabilities
        # require explicit registry evidence and are never hard-coded true.
        if any(s.kind == "audio" for s in specs.values()):
            advertised.add("audio_generation")
        if any(s.kind == "video" for s in specs.values()):
            advertised.add("video_generation")
        snapshot = build_capability_snapshot(provider_capabilities=advertised)
        return {
            **snapshot.capabilities,
            "evidence": snapshot.evidence,
            "async_jobs": True,
            "asset_uploads": True,
            "job_events": True,
            "named_models": len(specs),
            "presets": ["draft", "balanced", "quality", "max_quality"],
            "camera_presets": ["static", "dolly_in", "dolly_out", "pan_left", "pan_right", "orbit", "handheld", "drone"],
            "max_concurrency": max(1, int(os.getenv("GOPI_MEDIA_MAX_CONCURRENCY", "1"))),
            "job_workers": _job_workers,
            "max_queued_jobs": _job_queue_limit,
        }

    @router.get("/platform/resources")
    async def platform_resources(authorization: str | None = Header(default=None)):
        _require_auth(authorization)
        return {"object": "platform.resources", "gpus": gpu_report()}

    @router.post("/platform/route")
    async def platform_route(payload: dict[str, Any], authorization: str | None = Header(default=None)):
        _require_auth(authorization)
        try:
            decision = route_multimodal(
                input_modalities=payload.get("input_modalities", []),
                output_modality=payload.get("output_modality", "text"),
                operation=payload.get("operation", "generate"),
            )
        except Exception as error:
            raise HTTPException(422, str(error)) from error
        return {
            "object": "multimodal.route",
            "capability": decision.capability,
            "input_modalities": list(decision.input_modalities),
            "output_modality": decision.output_modality,
        }

    @router.post("/media/assets")
    async def upload_asset(request: Request, authorization: str | None = Header(default=None), x_filename: str | None = Header(default=None)):
        _require_auth(authorization)
        max_bytes = _asset_store().max_bytes
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > max_bytes:
            raise HTTPException(413, "asset is too large")
        body = await request.body()
        try:
            record = _asset_store().put(body, mime_type=request.headers.get("content-type", "application/octet-stream"), filename=x_filename)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        return {
            "id": record.id,
            "object": "media.asset",
            "filename": record.filename,
            "mime_type": record.mime_type,
            "size_bytes": record.size_bytes,
            "sha256": record.sha256,
            "created_at": record.created_at,
        }

    @router.get("/media/assets/{asset_id}")
    async def get_asset(asset_id: str, authorization: str | None = Header(default=None)):
        _require_auth(authorization)
        try:
            record = _asset_store().get(asset_id)
        except KeyError as error:
            raise HTTPException(404, "asset not found") from error
        return {
            "id": record.id,
            "object": "media.asset",
            "filename": record.filename,
            "mime_type": record.mime_type,
            "size_bytes": record.size_bytes,
            "sha256": record.sha256,
            "created_at": record.created_at,
        }

    @router.delete("/media/assets/{asset_id}")
    async def delete_asset(asset_id: str, authorization: str | None = Header(default=None)):
        _require_auth(authorization)
        if not _asset_store().delete(asset_id):
            raise HTTPException(404, "asset not found")
        return {"id": asset_id, "deleted": True}

    @router.post("/media/generations", status_code=202)
    async def create_generation(
        req: MediaGenerationRequest,
        authorization: str | None = Header(default=None),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ):
        _require_auth(authorization)
        if req.kind == "audio" and (req.init_image_asset_id or req.init_video_asset_id or req.scene_prompts or req.camera):
            raise HTTPException(422, "video-only fields were provided for an audio generation")
        if req.kind == "video" and req.init_audio_asset_id:
            raise HTTPException(422, "audio-only fields were provided for a video generation")
        if req.scene_prompts and len(req.scene_prompts) != req.segments:
            raise HTTPException(422, "scene_prompts length must equal segments")
        try:
            # Validate asset references before queueing.
            for asset_id in (req.init_audio_asset_id, req.init_image_asset_id, req.init_video_asset_id):
                if asset_id:
                    _asset_path(asset_id)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        try:
            return _job_dict(_submit_job(req, idempotency_key=idempotency_key))
        except _BusyError as error:
            raise HTTPException(429, str(error), headers={"Retry-After": "2"}) from error

    @router.get("/media/jobs")
    async def list_jobs(limit: int = 50, authorization: str | None = Header(default=None)):
        _require_auth(authorization)
        return {"object": "list", "data": [_job_dict(job) for job in _job_store().list(limit=limit)]}

    @router.get("/media/jobs/{job_id}")
    async def get_job(job_id: str, authorization: str | None = Header(default=None)):
        _require_auth(authorization)
        try:
            return _job_dict(_job_store().get(job_id))
        except KeyError as error:
            raise HTTPException(404, "media job not found") from error

    @router.post("/media/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str, authorization: str | None = Header(default=None)):
        _require_auth(authorization)
        try:
            job = _job_store().request_cancel(job_id)
        except KeyError as error:
            raise HTTPException(404, "media job not found") from error
        with _active_lock:
            token = _active_tokens.get(job_id)
            if token is not None:
                token.cancel()
        return _job_dict(job)

    @router.post("/media/jobs/{job_id}/retry", status_code=202)
    async def retry_job(job_id: str, authorization: str | None = Header(default=None)):
        _require_auth(authorization)
        try:
            old = _job_store().get(job_id)
        except KeyError as error:
            raise HTTPException(404, "media job not found") from error
        if old.status not in {"failed", "cancelled"}:
            raise HTTPException(409, "only failed or cancelled jobs can be retried")
        req = MediaGenerationRequest(**old.request)
        try:
            return _job_dict(_submit_job(req))
        except _BusyError as error:
            raise HTTPException(429, str(error), headers={"Retry-After": "2"}) from error

    @router.get("/media/jobs/{job_id}/content", response_class=FileResponse)
    async def job_content(job_id: str, authorization: str | None = Header(default=None)):
        _require_auth(authorization)
        try:
            job = _job_store().get(job_id)
        except KeyError as error:
            raise HTTPException(404, "media job not found") from error
        if job.status != "succeeded" or not job.result:
            raise HTTPException(409, f"media job is {job.status}")
        path = Path(str(job.result["path"]))
        if not path.exists():
            raise HTTPException(410, "generated artifact is no longer available")
        return FileResponse(path, media_type=str(job.result.get("media_type") or "application/octet-stream"), filename=path.name)

    @router.get("/media/jobs/{job_id}/events")
    async def job_events(job_id: str, authorization: str | None = Header(default=None)):
        _require_auth(authorization)
        try:
            _job_store().get(job_id)
        except KeyError as error:
            raise HTTPException(404, "media job not found") from error

        async def events():
            last = None
            while True:
                try:
                    job = _job_store().get(job_id)
                except KeyError:
                    yield "event: error\ndata: {\"error\":\"job not found\"}\n\n"
                    return
                payload = json.dumps(_job_dict(job), separators=(",", ":"))
                if payload != last:
                    yield f"event: job\ndata: {payload}\n\n"
                    last = payload
                if job.status in {"succeeded", "failed", "cancelled"}:
                    return
                await asyncio.sleep(0.5)

        return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    # Backward-compatible synchronous endpoints.
    @router.post("/audio/generations", response_class=FileResponse)
    async def audio(req: AudioGenerationRequest, authorization: str | None = Header(default=None)):
        _require_auth(authorization)
        out = Path(os.getenv("GOPI_MEDIA_OUTPUT_DIR", "outputs/api_media")) / f"audio-{uuid.uuid4().hex}.wav"
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            with _gate:
                path, _ = await run_in_threadpool(_run_audio, req, out)
        except _BusyError as error:
            raise HTTPException(429, str(error), headers={"Retry-After": "2"}) from error
        except (RuntimeError, FileNotFoundError) as error:
            raise HTTPException(503, str(error)) from error
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        return _response(path)

    @router.post("/video/generations", response_class=FileResponse)
    async def video(req: VideoGenerationRequest, authorization: str | None = Header(default=None)):
        _require_auth(authorization)
        if req.scene_prompts and len(req.scene_prompts) != req.segments:
            raise HTTPException(422, "scene_prompts length must equal segments")
        out = Path(os.getenv("GOPI_MEDIA_OUTPUT_DIR", "outputs/api_media")) / f"video-{uuid.uuid4().hex}.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            with _gate:
                path, _ = await run_in_threadpool(_run_video, req, out)
        except _BusyError as error:
            raise HTTPException(429, str(error), headers={"Retry-After": "2"}) from error
        except (RuntimeError, FileNotFoundError) as error:
            raise HTTPException(503, str(error)) from error
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        return _response(path)

    return router
