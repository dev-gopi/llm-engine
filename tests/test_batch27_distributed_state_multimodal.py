from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
import torch

from api.responses import ResponsesRequest
from image_data.dataset import CaptionedImageDataset
from omni_platform.native_multimodal import (
    CoquiXTTSVoiceCloningProvider,
    HuggingFaceAudioUnderstandingProvider,
    HuggingFaceVideoUnderstandingProvider,
    NativeLatentImageProvider,
)
from serving.realtime import RealtimeSession
from serving.redis_semantic_cache import RedisCacheConfig


def test_audio_only_responses_modality_is_valid():
    req = ResponsesRequest(model="gopi", input="hello", modalities=["audio"])
    assert req.modalities == ["audio"]


def test_optional_multimodal_providers_are_disabled_without_configuration(monkeypatch):
    for name in (
        "GOPI_AUDIO_UNDERSTANDING_MODEL", "GOPI_VIDEO_UNDERSTANDING_MODEL",
        "GOPI_VOICE_CLONING_MODEL", "GOPI_LATENT_IMAGE_CONFIG",
        "GOPI_LATENT_IMAGE_CHECKPOINT", "GOPI_LATENT_VAE_CHECKPOINT",
        "GOPI_LATENT_TEXT_ENCODER_CHECKPOINT", "GOPI_LATENT_IMAGE_TOKENIZER",
    ):
        monkeypatch.delenv(name, raising=False)
    assert HuggingFaceAudioUnderstandingProvider.from_env() is None
    assert HuggingFaceVideoUnderstandingProvider.from_env() is None
    assert CoquiXTTSVoiceCloningProvider.from_env() is None
    assert NativeLatentImageProvider.from_env() is None


def test_captioned_image_dataset_manifest(tmp_path: Path):
    from PIL import Image
    image = tmp_path / "x.png"
    Image.new("RGB", (16, 16), (1, 2, 3)).save(image)
    manifest = tmp_path / "train.jsonl"
    manifest.write_text(json.dumps({"image": "x.png", "text": "small test image"}) + "\n", encoding="utf-8")
    ds = CaptionedImageDataset(manifest, 16)
    tensor, caption = ds[0]
    assert tensor.shape == (3, 16, 16)
    assert caption == "small test image"


def test_redis_semantic_cache_enables_vector_index_by_default():
    cfg = RedisCacheConfig(url="redis://example.invalid:6379/0")
    assert cfg.vector_index_enabled is True
    assert cfg.vector_index_name


class _FakeResult:
    text = "answer"


class _FakeRuntime:
    async def generate(self, request):
        assert request.prompt == "hello"
        return _FakeResult()


def test_realtime_text_only_response_without_optional_audio_models():
    events = []
    async def send(payload): events.append(payload)
    async def run():
        session = RealtimeSession(_FakeRuntime(), send, model="gopi")
        session.modalities = ["text"]
        session.asr = None; session.tts = None
        await session.opened()
        await session.respond("hello")
    asyncio.run(run())
    types = [x["type"] for x in events]
    assert types == ["session.created", "response.created", "response.output_text.delta", "response.output_text.done", "response.done"]


def test_realtime_audio_commit_rejects_empty_buffer():
    events = []
    async def send(payload): events.append(payload)
    async def run():
        session = RealtimeSession(_FakeRuntime(), send, model="gopi")
        with pytest.raises(ValueError, match="audio buffer is empty"):
            await session.handle({"type": "input_audio_buffer.commit"})
    asyncio.run(run())


def test_native_latent_provider_advertises_image_editing():
    assert "image_editing" in NativeLatentImageProvider.capabilities


def test_native_latent_edit_validates_missing_source_before_runtime_load(tmp_path: Path):
    provider = NativeLatentImageProvider("cfg.yml", "model.pt", "vae.pt", "text.pt", "tok.json", output_dir=tmp_path)
    ctx = type("Ctx", (), {"request_id": "r1"})()
    # Replace runtime load so this test stays dependency/checkpoint independent.
    provider._load = lambda: ({"image_size": 16}, None, torch.device("cpu"), None)
    with pytest.raises(Exception, match="image file does not exist"):
        provider.edit_image({"prompt": "edit", "image": str(tmp_path / "missing.png")}, ctx)


class _PurgeRedis:
    def __init__(self):
        self.deleted=[]
    async def scan_iter(self, match=None, count=None):
        if ":entry:" in match:
            for x in ["p:entry:t:a", "p:entry:t:b"]: yield x
        elif ":vector:" in match:
            for x in ["p:vector:t:a", "p:vector:t:b"]: yield x
    async def delete(self, *keys):
        self.deleted.extend(keys); return len(keys)


def test_redis_namespace_purge_removes_entries_vectors_and_quota_membership():
    from serving.redis_semantic_cache import RedisSemanticCacheBackend
    async def run():
        fake=_PurgeRedis(); backend=RedisSemanticCacheBackend(RedisCacheConfig(key_prefix="p"),client=fake)
        assert await backend.purge("t") == 2
        assert set(fake.deleted) == {"p:entry:t:a","p:entry:t:b","p:vector:t:a","p:vector:t:b","p:quota-members:t"}
    asyncio.run(run())
