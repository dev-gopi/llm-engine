import pytest

from omni_platform.jobs import GenerationJob
from omni_platform.model_registry import OmniModelRecord, OmniModelRegistry


def test_generation_job_state_machine():
    job = GenerationJob(modality="video", model="video-x", provider="local")
    job.transition("initializing", progress=0.01)
    job.transition("processing", progress=0.2)
    job.transition("postprocessing", progress=0.95)
    job.transition("completed")
    assert job.progress == 1.0
    assert job.generation_duration is not None
    with pytest.raises(ValueError):
        job.transition("processing")


def test_model_registry_only_ready_models_advertise_capabilities():
    registry = OmniModelRegistry()
    registry.register(OmniModelRecord(
        id="video-x", family="gopi", provider="local", task="text-to-video",
        input_modalities=("text",), output_modalities=("video",),
        capabilities={"video_generation": True}, status="configured",
    ))
    assert "video_generation" not in registry.capabilities()
    registry.update_status("video-x", status="ready", loaded=True, device="cuda:0", precision="bf16")
    assert "video_generation" in registry.capabilities()
