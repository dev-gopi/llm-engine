from __future__ import annotations

import json
import wave

import pytest
import torch

from omni_platform.providers import ProviderContext
from omni_platform.voice_cloning import (
    CoquiXTTSVoiceCloningProvider,
    ProviderVoiceCloner,
    VoiceClonePolicy,
    VoiceCloneTrainer,
)


@pytest.fixture
def sample_wav(tmp_path):
    wav_path = tmp_path / "speaker.wav"
    sample_rate = 16000
    num_samples = 16000  # 1 second
    with wave.open(str(wav_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(b"\x00\x00" * num_samples)
    return wav_path


def test_coqui_xtts_provider_generation(sample_wav, tmp_path):
    provider = CoquiXTTSVoiceCloningProvider(output_dir=tmp_path / "audio_out")
    ctx = ProviderContext(request_id="req-123")
    res = provider.clone_voice(
        {
            "text": "Hello world from cloned voice",
            "speaker_wav": str(sample_wav),
            "language": "en",
        },
        ctx,
    )
    assert len(res.artifacts) == 1
    artifact = res.artifacts[0]
    assert artifact.path.is_file()
    assert artifact.mime_type == "audio/wav"
    assert artifact.metadata["provider"] == "coqui_xtts"
    assert artifact.metadata["language"] == "en"

    # Verify produced audio is a valid WAV file
    with wave.open(str(artifact.path), "rb") as w:
        assert w.getnchannels() == 1
        assert w.getframerate() == 16000
        assert w.getnframes() > 0


def test_voice_clone_trainer(sample_wav, tmp_path):
    trainer = VoiceCloneTrainer(output_root=tmp_path / "profiles")
    profile = trainer.train_voice_clone_adapter(
        [sample_wav],
        speaker_id="speaker_alice",
        learning_rate=2e-4,
        steps=5,
    )
    assert profile.speaker_id == "speaker_alice"
    assert profile.adapter_path.is_file()
    assert profile.reference_duration_seconds > 0

    checkpoint = torch.load(profile.adapter_path, weights_only=True)
    assert checkpoint["speaker_id"] == "speaker_alice"
    assert "speaker_embedding" in checkpoint
    assert checkpoint["speaker_embedding"].shape == (1, 512)

    profile_json = profile.adapter_path.parent / "profile.json"
    assert profile_json.is_file()
    data = json.loads(profile_json.read_text(encoding="utf-8"))
    assert data["speaker_id"] == "speaker_alice"


def test_provider_voice_cloner_policy_integration(sample_wav, tmp_path):
    provider = CoquiXTTSVoiceCloningProvider(output_dir=tmp_path / "out")
    policy = VoiceClonePolicy(require_consent_token=False, max_reference_seconds=30.0)
    cloner = ProviderVoiceCloner(provider, policy)
    ctx = ProviderContext(request_id="req-policy")
    res = cloner.clone_voice(
        {
            "text": "Testing voice cloning policy wrapper",
            "speaker_wav": str(sample_wav),
            "reference_duration_seconds": 2.0,
        },
        ctx,
    )
    assert len(res.artifacts) == 1
    assert res.artifacts[0].path.is_file()
