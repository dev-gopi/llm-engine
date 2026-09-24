from pathlib import Path

import pytest

from omni_platform.artifacts import LocalArtifactStorage
from omni_platform.capabilities import build_capability_snapshot
from omni_platform.errors import UnsupportedCapabilityError
from omni_platform.providers import ProviderRegistry
from omni_platform.queue import InProcessJobQueue, QueueMessage
from omni_platform.router import route_multimodal
from omni_platform.validation import GenerationLimits, validate_video


class FakeProvider:
    id = "video-test"
    capabilities = frozenset({"video_generation", "image_to_video"})

    def is_available(self):
        return True


def test_provider_registry_and_capability_snapshot():
    registry = ProviderRegistry()
    registry.register(FakeProvider())
    snapshot = build_capability_snapshot(provider_capabilities=registry.capability_set())
    assert snapshot.enabled("video_generation") is True
    assert snapshot.enabled("image_to_video") is True
    assert snapshot.enabled("speech_to_text") is False


def test_router_is_deterministic():
    decision = route_multimodal(
        input_modalities=["text", "image"], output_modality="video", operation="generate"
    )
    assert decision.capability == "image_to_video"
    with pytest.raises(UnsupportedCapabilityError):
        route_multimodal(input_modalities=["video"], output_modality="audio", operation="generate")


def test_local_artifact_storage_does_not_expose_source_path(tmp_path: Path):
    src = tmp_path / "sample.wav"
    src.write_bytes(b"RIFFxxxxWAVE")
    store = LocalArtifactStorage(tmp_path / "store")
    record = store.create_record(src, job_id="job1", type="audio")
    store.put(src, record)
    resolved = store.resolve(record.id)
    assert resolved.is_file()
    assert str(src) not in record.storage_key
    assert record.checksum_sha256
    store.delete(record.id)
    assert not resolved.exists()


def test_bounded_priority_queue_and_dead_letter():
    q = InProcessJobQueue(maxsize=2)
    q.put(QueueMessage("low", {}, priority=0))
    q.put(QueueMessage("high", {}, priority=10))
    assert q.get(timeout=0.1).id == "high"
    msg = q.get(timeout=0.1)
    q.dead_letter(msg, "failed")
    assert q.dead_letters[0][0].id == "low"


def test_video_limits():
    limits = GenerationLimits(max_video_seconds=10, max_video_fps=30, max_video_frames=300)
    validate_video(frames=300, fps=30, limits=limits)
    with pytest.raises(Exception):
        validate_video(frames=301, fps=30, limits=limits)
