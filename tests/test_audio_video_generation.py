from __future__ import annotations

import wave

import torch

from audio_generation.io import load_wav, save_wav
from audio_generation.model import AudioDiffusionModel
from audio_generation.pipeline import AudioGenerationPipeline
from diffusion.scheduler import DiffusionScheduler
from diffusion.text_encoder import DiffusionTextEncoder
from media_generation.conditioning import TextConditioner
from video_generation.model import VideoDiffusionModel
from video_generation.pipeline import VideoGenerationPipeline


def _conditioner(hidden: int = 32) -> TextConditioner:
    return TextConditioner(DiffusionTextEncoder(128, hidden_size=hidden, layers=1, heads=4, max_length=16))


def test_audio_autoencoder_and_denoiser_shapes():
    model = AudioDiffusionModel.from_config(
        {"autoencoder_channels": 8, "latent_channels": 4, "downsample_stages": 3, "model_channels": 16, "blocks": 2},
        text_conditioner=_conditioner(),
    )
    waveform = torch.randn(2, 320)
    reconstruction, latents = model.autoencoder(waveform)
    assert reconstruction.shape == (2, 1, 320)
    prediction = model.denoiser(latents, torch.tensor([1, 2]), torch.randn(2, 32))
    assert prediction.shape == latents.shape


def test_audio_pipeline_loss_and_short_sample():
    model = AudioDiffusionModel.from_config(
        {"autoencoder_channels": 8, "latent_channels": 4, "downsample_stages": 2, "model_channels": 16, "blocks": 1},
        text_conditioner=_conditioner(),
    )
    scheduler = DiffusionScheduler(8, schedule="cosine")
    pipeline = AudioGenerationPipeline(model, scheduler)
    condition = torch.randn(1, 32)
    loss, metrics = pipeline.training_loss(torch.randn(1, 128), condition)
    assert torch.isfinite(loss)
    assert set(metrics) == {"diffusion", "reconstruction"}
    sample = pipeline.sample(condition, samples=64, sample_rate=16000, device="cpu", inference_steps=2, seed=1)
    assert sample.shape == (1, 64)
    assert torch.isfinite(sample).all()


def test_wav_roundtrip(tmp_path):
    path = save_wav(tmp_path / "test.wav", torch.linspace(-0.5, 0.5, 160), 16000)
    with wave.open(str(path), "rb") as handle:
        assert handle.getframerate() == 16000
        assert handle.getnchannels() == 1
    loaded = load_wav(path, sample_rate=16000, samples=200)
    assert loaded.shape == (200,)


def test_video_autoencoder_and_denoiser_shapes():
    model = VideoDiffusionModel.from_config(
        {"autoencoder_channels": 8, "latent_channels": 4, "spatial_stages": 2, "model_channels": 16, "blocks": 2, "temporal_heads": 4, "attention_every": 1},
        text_conditioner=_conditioner(),
    )
    video = torch.randn(1, 3, 4, 32, 32)
    reconstruction, latents = model.autoencoder(video)
    assert reconstruction.shape == video.shape
    assert latents.shape[-3:] == (2, 8, 8)
    prediction = model.denoiser(latents, torch.tensor([1]), torch.randn(1, 32))
    assert prediction.shape == latents.shape


def test_video_pipeline_loss_and_short_sample():
    model = VideoDiffusionModel.from_config(
        {"autoencoder_channels": 8, "latent_channels": 4, "spatial_stages": 2, "model_channels": 16, "blocks": 1, "temporal_heads": 4, "attention_every": 1},
        text_conditioner=_conditioner(),
    )
    scheduler = DiffusionScheduler(8, schedule="cosine")
    pipeline = VideoGenerationPipeline(model, scheduler)
    condition = torch.randn(1, 32)
    loss, metrics = pipeline.training_loss(torch.randn(1, 3, 4, 16, 16), condition)
    assert torch.isfinite(loss)
    assert set(metrics) == {"diffusion", "reconstruction"}
    sample = pipeline.sample(condition, frames=2, height=16, width=16, device="cpu", inference_steps=2, seed=1)
    assert sample.shape == (1, 3, 2, 16, 16)
    assert torch.isfinite(sample).all()


def test_audio_cross_attention_and_init_audio_sampling():
    model = AudioDiffusionModel.from_config(
        {
            "autoencoder_channels": 8,
            "latent_channels": 4,
            "downsample_stages": 2,
            "model_channels": 16,
            "blocks": 2,
            "cross_attention_every": 1,
            "cross_attention_heads": 4,
        },
        text_conditioner=_conditioner(),
    )
    scheduler = DiffusionScheduler(8, schedule="cosine")
    pipeline = AudioGenerationPipeline(model, scheduler)
    condition = torch.randn(1, 32)
    context = torch.randn(1, 5, 32)
    mask = torch.ones(1, 5, dtype=torch.bool)
    init = torch.randn(1, 64).clamp(-1, 1)
    sample = pipeline.sample(
        condition,
        samples=64,
        sample_rate=16000,
        device="cpu",
        context=context,
        context_mask=mask,
        inference_steps=2,
        seed=1,
        init_waveform=init,
        strength=0.5,
    )
    assert sample.shape == (1, 64)
    assert torch.isfinite(sample).all()


def test_video_cross_attention_and_init_video_sampling():
    model = VideoDiffusionModel.from_config(
        {
            "autoencoder_channels": 8,
            "latent_channels": 4,
            "spatial_stages": 2,
            "model_channels": 16,
            "blocks": 2,
            "temporal_heads": 4,
            "attention_every": 1,
            "cross_attention_every": 1,
            "cross_attention_heads": 4,
            "cross_attention_chunk_size": 16,
        },
        text_conditioner=_conditioner(),
    )
    scheduler = DiffusionScheduler(8, schedule="cosine")
    pipeline = VideoGenerationPipeline(model, scheduler)
    condition = torch.randn(1, 32)
    context = torch.randn(1, 5, 32)
    mask = torch.ones(1, 5, dtype=torch.bool)
    init = torch.randn(1, 3, 2, 16, 16).clamp(-1, 1)
    sample = pipeline.sample(
        condition,
        frames=2,
        height=16,
        width=16,
        device="cpu",
        context=context,
        context_mask=mask,
        inference_steps=2,
        seed=1,
        init_video=init,
        strength=0.5,
    )
    assert sample.shape == (1, 3, 2, 16, 16)
    assert torch.isfinite(sample).all()


def test_min_snr_training_loss_is_finite():
    model = AudioDiffusionModel.from_config(
        {"autoencoder_channels": 8, "latent_channels": 4, "downsample_stages": 2, "model_channels": 16, "blocks": 1},
        text_conditioner=_conditioner(),
    )
    pipeline = AudioGenerationPipeline(model, DiffusionScheduler(8, schedule="cosine"))
    loss, _ = pipeline.training_loss(
        torch.randn(2, 128), torch.randn(2, 32), min_snr_gamma=5.0, noise_offset=0.01, input_perturbation=0.02
    )
    assert torch.isfinite(loss)


def test_audio_preserve_mask_and_progress_callback():
    model = AudioDiffusionModel.from_config(
        {"autoencoder_channels": 8, "latent_channels": 4, "downsample_stages": 2, "model_channels": 16, "blocks": 1},
        text_conditioner=_conditioner(),
    )
    pipeline = AudioGenerationPipeline(model, DiffusionScheduler(8, schedule="cosine"))
    seen=[]
    out = pipeline.sample(
        torch.randn(1, 32), samples=64, sample_rate=16000, device="cpu", inference_steps=2, seed=1,
        init_waveform=torch.randn(1,64), strength=.5, preserve_mask=torch.cat([torch.ones(32),torch.zeros(32)]),
        progress_callback=lambda current,total: seen.append((current,total)),
    )
    assert out.shape == (1,64)
    assert seen and seen[-1][0] == seen[-1][1]


def test_video_preserve_mask_and_progress_callback():
    model = VideoDiffusionModel.from_config(
        {"autoencoder_channels":8,"latent_channels":4,"spatial_stages":2,"model_channels":16,"blocks":1,"temporal_heads":4,"attention_every":1},
        text_conditioner=_conditioner(),
    )
    pipeline=VideoGenerationPipeline(model,DiffusionScheduler(8,schedule="cosine")); seen=[]
    init=torch.randn(1,3,2,16,16)
    mask=torch.zeros(2,16,16); mask[:,:, :8]=1
    out=pipeline.sample(torch.randn(1,32),frames=2,height=16,width=16,device="cpu",inference_steps=2,seed=1,init_video=init,strength=.5,preserve_mask=mask,progress_callback=lambda c,t: seen.append((c,t)))
    assert out.shape==(1,3,2,16,16)
    assert seen and seen[-1][0] == seen[-1][1]
