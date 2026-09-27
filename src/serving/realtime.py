"""Realtime multimodal event API over WebSocket and optional WebRTC data channels.

The transport accepts OpenAI-style JSON events. Audio buffer payloads are base64
encoded WAV bytes; this keeps codec handling explicit and avoids pretending raw
PCM formats are interchangeable. WebRTC support is optional through ``aiortc``.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import secrets
import tempfile
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable

from fastapi import (
    APIRouter,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from pydantic import BaseModel, ConfigDict, Field

from omni_platform.providers import ProviderContext
from omni_platform.speech import HuggingFaceASRProvider, HuggingFaceTTSProvider
from serving.schemas import GenerateRequest

router = APIRouter(tags=["realtime"])
Send = Callable[[dict[str, Any]], Awaitable[None]]


class WebRTCOffer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sdp: str = Field(min_length=1, max_length=1_000_000)
    type: str = Field(default="offer", pattern="^offer$")


def _authorized(headers, settings) -> bool:
    if not settings.api_key:
        return True
    supplied = headers.get("authorization", "").removeprefix("Bearer ")
    protocols = [
        x.strip() for x in headers.get("sec-websocket-protocol", "").split(",")
    ]
    if len(protocols) >= 2 and protocols[0].lower() == "bearer":
        supplied = protocols[1]
    return bool(supplied and secrets.compare_digest(supplied, settings.api_key))


class RealtimeSession:
    def __init__(self, runtime: Any, send: Send, *, model: str) -> None:
        self.runtime = runtime
        self.send = send
        self.model = model
        self.id = f"sess_{uuid.uuid4().hex}"
        self.audio = bytearray()
        self.modalities = ["text", "audio"]
        self.instructions = ""
        self.temperature = 0.7
        self.max_tokens = 256
        self.asr = HuggingFaceASRProvider.from_env()
        self.tts = HuggingFaceTTSProvider.from_env()
        self.last_transcript = ""

    async def opened(self) -> None:
        await self.send(
            {
                "type": "session.created",
                "session": {
                    "id": self.id,
                    "model": self.model,
                    "modalities": self.modalities,
                },
            }
        )

    async def handle(self, event: dict[str, Any]) -> None:
        kind = str(event.get("type", ""))
        if kind == "session.update":
            session = event.get("session") or {}
            mods = session.get("modalities")
            if (
                isinstance(mods, list)
                and mods
                and all(x in {"text", "audio"} for x in mods)
            ):
                self.modalities = list(dict.fromkeys(mods))
            self.instructions = str(session.get("instructions") or self.instructions)
            self.temperature = float(session.get("temperature", self.temperature))
            self.max_tokens = int(session.get("max_output_tokens", self.max_tokens))
            await self.send(
                {
                    "type": "session.updated",
                    "session": {
                        "id": self.id,
                        "model": self.model,
                        "modalities": self.modalities,
                    },
                }
            )
            return
        if kind == "input_audio_buffer.clear":
            self.audio.clear()
            await self.send({"type": "input_audio_buffer.cleared"})
            return
        if kind == "input_audio_buffer.append":
            chunk = event.get("audio")
            if not isinstance(chunk, str):
                raise ValueError("audio must be base64 text")
            raw = base64.b64decode(chunk, validate=True)
            if len(self.audio) + len(raw) > int(
                os.getenv("GOPI_REALTIME_MAX_AUDIO_BYTES", str(25 * 1024 * 1024))
            ):
                raise ValueError("realtime audio buffer exceeds limit")
            self.audio.extend(raw)
            return
        if kind == "input_audio_buffer.commit":
            if not self.audio:
                raise ValueError("audio buffer is empty")
            if self.asr is None or not self.asr.is_available():
                raise RuntimeError("ASR provider is unavailable")
            with tempfile.NamedTemporaryFile(
                prefix="gopi-realtime-", suffix=".wav", delete=False
            ) as fh:
                fh.write(self.audio)
                path = Path(fh.name)
            self.audio.clear()
            try:
                result = await asyncio.to_thread(
                    self.asr.transcribe,
                    {"path": str(path)},
                    ProviderContext(request_id=f"rtasr_{uuid.uuid4().hex}"),
                )
            finally:
                path.unlink(missing_ok=True)
            self.last_transcript = str(result.get("text", "")).strip()
            await self.send(
                {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "transcript": self.last_transcript,
                }
            )
            if event.get("create_response", True):
                await self.respond(self.last_transcript)
            return
        if kind == "conversation.item.create":
            item = event.get("item") or {}
            text = str(item.get("text") or item.get("content") or "").strip()
            if text:
                self.last_transcript = text
            await self.send(
                {
                    "type": "conversation.item.created",
                    "item": {"role": "user", "text": text},
                }
            )
            return
        if kind == "response.create":
            response = event.get("response") or {}
            text = str(response.get("input") or self.last_transcript).strip()
            await self.respond(text)
            return
        if kind == "response.cancel":
            await self.send({"type": "response.cancelled"})
            return
        raise ValueError(f"unsupported realtime event type: {kind}")

    async def respond(self, text: str) -> None:
        if not text:
            raise ValueError("response input is empty")
        prompt = f"{self.instructions}\n{text}".strip() if self.instructions else text
        rid = f"resp_{uuid.uuid4().hex}"
        await self.send(
            {
                "type": "response.created",
                "response": {"id": rid, "status": "in_progress"},
            }
        )
        result = await self.runtime.generate(
            GenerateRequest(
                prompt=prompt,
                max_tokens=max(1, min(self.max_tokens, 8192)),
                temperature=max(0.0, min(self.temperature, 2.0)),
            )
        )
        if "text" in self.modalities:
            await self.send(
                {
                    "type": "response.output_text.delta",
                    "response_id": rid,
                    "delta": result.text,
                }
            )
            await self.send(
                {
                    "type": "response.output_text.done",
                    "response_id": rid,
                    "text": result.text,
                }
            )
        if "audio" in self.modalities:
            if self.tts is None or not self.tts.is_available():
                raise RuntimeError("TTS provider is unavailable")
            audio = await asyncio.to_thread(
                self.tts.synthesize,
                {"text": result.text},
                ProviderContext(request_id=f"rttts_{uuid.uuid4().hex}"),
            )
            raw = audio.artifacts[0].path.read_bytes()
            await self.send(
                {
                    "type": "response.audio.delta",
                    "response_id": rid,
                    "audio": base64.b64encode(raw).decode("ascii"),
                    "format": "wav",
                }
            )
            await self.send({"type": "response.audio.done", "response_id": rid})
        await self.send(
            {"type": "response.done", "response": {"id": rid, "status": "completed"}}
        )


@router.websocket("/v1/realtime")
async def realtime_ws(websocket: WebSocket) -> None:
    settings = websocket.app.state.settings
    if not _authorized(websocket.headers, settings):
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    await websocket.accept()

    async def send(payload):
        await websocket.send_json(payload)

    session = RealtimeSession(
        websocket.app.state.runtime, send, model=settings.model_name
    )
    await session.opened()
    while True:
        try:
            event = await websocket.receive_json()
            try:
                await session.handle(event)
            except Exception as exc:
                await send(
                    {
                        "type": "error",
                        "error": {"code": "realtime_event_error", "message": str(exc)},
                    }
                )
        except WebSocketDisconnect:
            return


_PEERS: set[Any] = set()


@router.post("/v1/realtime/webrtc")
async def realtime_webrtc(offer: WebRTCOffer, request: Request):
    settings = request.app.state.settings
    if not _authorized(request.headers, settings):
        raise HTTPException(401, "invalid bearer token")
    try:
        from aiortc import RTCPeerConnection, RTCSessionDescription
    except ImportError as exc:
        raise HTTPException(
            503, "WebRTC requires the optional aiortc dependency"
        ) from exc
    pc = RTCPeerConnection()
    _PEERS.add(pc)

    @pc.on("connectionstatechange")
    async def state_change():
        if pc.connectionState in {"failed", "closed", "disconnected"}:
            await pc.close()
            _PEERS.discard(pc)

    @pc.on("datachannel")
    def datachannel(channel):
        async def send(payload):
            if channel.readyState == "open":
                channel.send(json.dumps(payload, separators=(",", ":")))

        session = RealtimeSession(
            request.app.state.runtime, send, model=settings.model_name
        )

        @channel.on("open")
        def opened():
            asyncio.create_task(session.opened())

        @channel.on("message")
        def message(data):
            if isinstance(data, str):
                try:
                    payload = json.loads(data)
                except Exception:
                    payload = {"type": "invalid"}
                asyncio.create_task(session.handle(payload))

    await pc.setRemoteDescription(RTCSessionDescription(sdp=offer.sdp, type="offer"))
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}


__all__ = ["router", "RealtimeSession"]
