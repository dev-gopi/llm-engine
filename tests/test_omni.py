from __future__ import annotations

import asyncio
import io
import wave
from pathlib import Path

import numpy as np
import pytest
from fastapi import FastAPI

from api.responses import ResponsesRequest
from media_generation.assets import AssetStore
from omni_platform.multimodal_input import latest_text, prepare_responses_input
from omni_platform.observability import Metrics
from omni_platform.scheduler import GPUScheduler, ResourceRequest
from omni_platform.speech import EnergyVAD, HuggingFaceASRProvider
from omni_platform.video_understanding import _sample_times
from serving.omni import create_omni_v6_router
from tests.asgi_client import ASGIClient


def _wav(path: Path, *, rate: int = 16000) -> None:
    silence = np.zeros(rate // 4, dtype=np.int16)
    t = np.arange(rate // 2)
    tone = (np.sin(2 * np.pi * 440 * t / rate) * 12000).astype(np.int16)
    data = np.concatenate([silence, tone, silence])
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(data.tobytes())


def _wav_bytes() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 1600)
    return buffer.getvalue()


class FakeASR:
    def is_available(self) -> bool:
        return True

    def transcribe(self, request, context):
        return {"text": "hello from audio"}


def test_energy_vad_detects_tone(tmp_path: Path) -> None:
    path = tmp_path / "sample.wav"
    _wav(path)
    segments = EnergyVAD().detect(path, threshold_dbfs=-35, min_speech_ms=100)
    assert len(segments) == 1
    assert 0.15 <= segments[0]["start"] <= 0.35
    assert 0.65 <= segments[0]["end"] <= 0.9


def test_sample_times_are_bounded_and_even() -> None:
    assert _sample_times(0, 5) == [0.0]
    values = _sample_times(10, 3)
    assert len(values) == 3 and 0 <= values[0] < values[1] < values[2] <= 10


def test_metrics_prometheus_output() -> None:
    metrics = Metrics()
    metrics.inc("requests_total", kind="audio")
    metrics.set("gpu_memory_bytes", 123, gpu="0")
    text = metrics.prometheus()
    assert 'gopi_requests_total{kind="audio"} 1.0' in text
    assert 'gopi_gpu_memory_bytes{gpu="0"} 123' in text


def test_gpu_scheduler_cpu_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("omni_platform.scheduler.gpu_report", lambda: [])
    assert GPUScheduler().select(ResourceRequest(allow_cpu=True)) == "cpu"


def test_gpu_scheduler_rejects_missing_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("omni_platform.scheduler.gpu_report", lambda: [])
    with pytest.raises(Exception):
        GPUScheduler().select(ResourceRequest(allow_cpu=False))


def test_asr_env_absent_is_not_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOPI_ASR_MODEL", raising=False)
    assert HuggingFaceASRProvider.from_env() is None


def test_omni_requires_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOPI_API_KEY", "secret")
    app = FastAPI()
    app.include_router(create_omni_v6_router())
    with ASGIClient(app) as client:
        assert client.get("/v1/platform/capabilities/verify").status_code == 401
        result = client.get(
            "/v1/platform/capabilities/verify", headers={"Authorization": "Bearer secret"}
        )
        assert result.status_code == 200
        assert result.json()["capabilities"]["speech_to_text"] is False


def test_tts_endpoint_conservative_without_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOPI_API_KEY", "secret")
    monkeypatch.delenv("GOPI_TTS_MODEL", raising=False)
    app = FastAPI()
    app.include_router(create_omni_v6_router())
    with ASGIClient(app) as client:
        result = client.post(
            "/v1/audio/speech", headers={"Authorization": "Bearer secret"}, json={"input": "hello"}
        )
        assert result.status_code == 503


def test_responses_accepts_typed_multimodal_input() -> None:
    request = ResponsesRequest.model_validate({
        "model": "gopi",
        "input": [{
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Describe this"},
                {"type": "input_image", "image_url": "data:image/png;base64,iVBORw0KGgo="},
            ],
        }],
    })
    assert request.input[0].content[0].type == "input_text"
    assert request.input[0].content[1].type == "input_image"


def test_responses_image_requires_exactly_one_source() -> None:
    with pytest.raises(ValueError):
        ResponsesRequest.model_validate({
            "model": "gopi",
            "input": [{
                "role": "user",
                "content": [{
                    "type": "input_image",
                    "image_url": "data:image/png;base64,iVBORw0KGgo=",
                    "asset_id": "asset_0123456789abcdef0123456789abcdef",
                }],
            }],
        })


def test_prepare_image_asset_and_audio_transcript(tmp_path: Path) -> None:
    store = AssetStore(tmp_path / "assets")
    image = store.put(b"\x89PNG\r\n\x1a\n" + b"payload", mime_type="image/png", filename="x.png")
    audio = store.put(_wav_bytes(), mime_type="audio/wav", filename="x.wav")
    request = ResponsesRequest.model_validate({
        "model": "gopi",
        "input": [{
            "role": "user",
            "content": [
                {"type": "input_text", "text": "What is here?"},
                {"type": "input_image", "asset_id": image.id},
                {"type": "input_audio", "asset_id": audio.id},
            ],
        }],
    })
    prepared = asyncio.run(prepare_responses_input(request.input, asset_store=store, asr=FakeASR()))
    content = prepared.messages[0]["content"]
    assert content[0]["text"] == "What is here?"
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert "hello from audio" in content[2]["text"]
    assert prepared.metadata["image_inputs"] == 1
    assert prepared.metadata["audio_inputs"][0]["transcribed"] is True
    assert latest_text(prepared.messages)


def test_audio_output_modality_requires_text_too() -> None:
    with pytest.raises(ValueError):
        ResponsesRequest.model_validate({"model": "gopi", "input": "hi", "modalities": ["audio"]})
