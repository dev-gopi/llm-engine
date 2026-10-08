from __future__ import annotations

import io
import json
import wave

import pytest

from omni_platform.audio_responses import (
    chunk_audio_stream,
    encode_pcm_to_wav,
    validate_audio_only_request,
)
from omni_platform.provider_adapters import RemoteJSONProvider, RemoteProviderConfig
from omni_platform.providers import ProviderContext


def test_audio_encoding_and_chunking():
    # Generate 16000 samples of 16-bit PCM (32000 bytes = 1 sec at 16kHz)
    pcm = b"\x00\x00" * 16000
    wav_bytes = encode_pcm_to_wav(pcm, channels=1, sample_width=2, frame_rate=16000)
    assert len(wav_bytes) > len(pcm)
    assert wav_bytes[:4] == b"RIFF"

    # Verify with wave module
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        assert w.getnchannels() == 1
        assert w.getframerate() == 16000
        assert w.getnframes() == 16000

    chunks = list(
        chunk_audio_stream(
            wav_bytes, chunk_size_bytes=8192, format="wav", sample_rate=16000
        )
    )
    assert len(chunks) > 1
    assert chunks[0].chunk_index == 0
    assert not chunks[0].is_final
    assert chunks[-1].is_final
    assert b"".join(c.audio for c in chunks) == wav_bytes


def test_validate_audio_only_request():
    with pytest.raises(ValueError, match="audio-only response requires audio input"):
        validate_audio_only_request({})

    with pytest.raises(ValueError, match="must not require text output"):
        validate_audio_only_request(
            {"audio": "dummy", "text": "not allowed", "audio_only": True}
        )

    validate_audio_only_request({"audio": "dummy", "audio_only": True})


def test_remote_json_provider_multimodal_methods(tmp_path):
    class MockTransport:
        def __init__(self, req, timeout):
            self.req = req

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            data = json.loads(self.req.data.decode("utf-8"))
            op = data["operation"]
            return json.dumps(
                {
                    "artifacts": [
                        {
                            "path": f"/tmp/{op}.out",
                            "mime_type": "application/octet-stream",
                            "metadata": {"status": "ok"},
                        }
                    ],
                    "usage": {"duration": 1.5},
                }
            ).encode("utf-8")

    cfg = RemoteProviderConfig(
        id="mock_vendor",
        endpoint="https://api.vendor.ai/v1/generate",
        capabilities=frozenset(
            {
                "image_generation",
                "audio_generation",
                "video_generation",
                "audio_understanding",
            }
        ),
        api_key="secret-key",
    )
    provider = RemoteJSONProvider(cfg, transport=MockTransport)
    ctx = ProviderContext(request_id="test-req")

    img_res = provider.generate_image({"prompt": "sunset"}, ctx)
    assert len(img_res.artifacts) == 1
    assert "image_generation" in str(img_res.artifacts[0].path)

    audio_res = provider.generate_audio({"prompt": "guitar music"}, ctx)
    assert len(audio_res.artifacts) == 1

    video_res = provider.generate_video({"prompt": "ocean waves"}, ctx)
    assert len(video_res.artifacts) == 1
