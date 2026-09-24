import pytest

from media_generation.production import optimizer_steps_per_epoch, validate_generation_config
from video_generation.io import _sample_indices


def test_optimizer_steps_uses_ceiling_for_partial_accumulation_group():
    assert optimizer_steps_per_epoch(10, 4) == 3


def test_video_contiguous_sampling_keeps_temporal_neighbors():
    indices = _sample_indices(20, 5, "center_contiguous")
    assert indices == [7, 8, 9, 10, 11]


def test_generation_config_rejects_bad_video_divisibility():
    config = {
        "tokenizer": "tok",
        "train_manifest": "train.jsonl",
        "frames": 8,
        "height": 70,
        "width": 64,
        "fps": 8,
        "spatial_stages": 3,
        "model_channels": 96,
        "temporal_heads": 4,
        "cross_attention_heads": 4,
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "epochs": 1,
        "timesteps": 1000,
        "learning_rate": 1e-4,
        "max_grad_norm": 1.0,
    }
    with pytest.raises(ValueError, match="divisible"):
        validate_generation_config(config, kind="video")

from media_generation.presets import apply_style, resolve_preset
from audio_generation.longform import crossfade_chunks, plan_windows
from video_generation.longform import blend_video_segments
from media_generation.quality import audio_diagnostics, video_diagnostics
import torch


def test_inference_presets_and_style_helpers():
    preset = resolve_preset("quality", {})
    assert preset.steps == 70
    assert "professional lighting" in apply_style("a city", "cinematic")


def test_audio_longform_planning_and_crossfade():
    assert len(plan_windows(1000, 400, 100)) == 3
    out = crossfade_chunks([torch.ones(10), torch.zeros(10)], 4)
    assert out.numel() == 16
    assert torch.isfinite(out).all()


def test_video_segment_blending():
    a = torch.zeros(3, 4, 4, 4)
    b = torch.ones(3, 4, 4, 4)
    out = blend_video_segments([a, b], 2)
    assert out.shape == (3, 6, 4, 4)
    assert torch.isfinite(out).all()


def test_media_quality_diagnostics():
    a = audio_diagnostics(torch.tensor([0.0, 0.5, -0.5]))
    assert 0 <= a["clipping_fraction"] <= 1
    v = video_diagnostics(torch.zeros(3, 2, 4, 4))
    assert v["temporal_abs_delta"] == 0

from media_generation.assets import AssetStore
from media_generation.jobs import MediaJobStore
from media_generation.storyboard import build_storyboard
from media_generation.tooling import MEDIA_TOOL_SCHEMAS


def test_asset_store_is_content_addressed_and_safe(tmp_path):
    store = AssetStore(tmp_path, max_bytes=1024)
    record = store.put(b"\x89PNG\r\n\x1a\n" + b"fake", mime_type="image/png", filename="../../unsafe.png")
    assert record.id.startswith("asset_")
    assert store.resolve(record.id).parent == tmp_path.resolve()
    assert store.get(record.id).sha256 == record.sha256
    assert store.delete(record.id) is True


def test_job_store_lifecycle_and_cancel(tmp_path):
    store = MediaJobStore(tmp_path / "jobs.sqlite3")
    job = store.create("video", {"kind": "video", "prompt": "hello"})
    assert job.status == "queued"
    job = store.update(job.id, status="running", progress=0.5)
    assert job.progress == 0.5
    job = store.request_cancel(job.id)
    assert job.cancel_requested is True
    job = store.update(job.id, status="cancelled", progress=0.0)
    assert job.status == "cancelled"


def test_storyboard_camera_and_scene_validation():
    shots = build_storyboard("base", ["scene one", "scene two"], camera="orbit")
    assert len(shots) == 2
    assert "orbit" in shots[0].prompt
    with pytest.raises(ValueError, match="camera"):
        build_storyboard("base", camera="teleport")


def test_media_agent_tool_schemas_are_strict():
    assert MEDIA_TOOL_SCHEMAS["generate_audio"]["additionalProperties"] is False
    assert "camera" in MEDIA_TOOL_SCHEMAS["generate_video"]["properties"]

def test_job_store_idempotency(tmp_path):
    store = MediaJobStore(tmp_path / "jobs.sqlite3")
    first, created_first = store.create_or_get(
        "audio", {"kind": "audio", "prompt": "one"}, idempotency_key="same-request"
    )
    second, created_second = store.create_or_get(
        "audio", {"kind": "audio", "prompt": "two"}, idempotency_key="same-request"
    )
    assert created_first is True
    assert created_second is False
    assert first.id == second.id
    assert second.request["prompt"] == "one"
