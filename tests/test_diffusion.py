import pytest
import torch

from diffusion.pipeline import DiffusionPipeline
from diffusion.scheduler import DiffusionScheduler
from diffusion.unet import SmallUNet
from diffusion.vae import AutoencoderKL
from diffusion.text_encoder import DiffusionTextEncoder
from diffusion.latent_pipeline import LatentDiffusionPipeline


def test_scheduler_add_noise_preserves_shape_and_known_endpoints() -> None:
    scheduler = DiffusionScheduler(timesteps=10)
    images = torch.ones(2, 3, 8, 8)
    noise = torch.zeros_like(images)
    noisy, returned_noise = scheduler.add_noise(images, torch.tensor([0, 9]), noise)
    assert noisy.shape == images.shape
    assert torch.equal(returned_noise, noise)
    assert torch.all(noisy <= images)
    assert torch.all(noisy > 0)


def test_small_unet_predicts_image_shaped_noise() -> None:
    model = SmallUNet(image_channels=3, base_channels=8, condition_size=16)
    output = model(
        torch.randn(2, 3, 16, 16),
        torch.tensor([1, 5]),
        torch.randn(2, 16),
    )
    assert output.shape == (2, 3, 16, 16)
    assert torch.isfinite(output).all()


def test_diffusion_training_loss_is_finite_and_differentiable() -> None:
    model = SmallUNet(image_channels=3, base_channels=8, condition_size=16)
    pipeline = DiffusionPipeline(model, DiffusionScheduler(timesteps=10))
    loss = pipeline.training_loss(torch.randn(2, 3, 16, 16), torch.randn(2, 16))
    loss.backward()
    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_diffusion_sampling_runs_complete_reverse_process() -> None:
    model = SmallUNet(image_channels=3, base_channels=8, condition_size=16)
    pipeline = DiffusionPipeline(model, DiffusionScheduler(timesteps=4))
    sample = pipeline.sample(1, 8, device="cpu")
    assert sample.shape == (1, 3, 8, 8)
    assert sample.min() >= -1
    assert sample.max() <= 1


def test_diffusion_rejects_incompatible_image_size() -> None:
    model = SmallUNet(image_channels=3, base_channels=8, condition_size=16)
    with pytest.raises(ValueError, match="divisible by four"):
        model(torch.randn(1, 3, 10, 10), torch.tensor([1]))


def test_cosine_scheduler_and_reduced_ddim_sampling() -> None:
    model = SmallUNet(image_channels=3, base_channels=8, condition_size=16,
                      use_attention=True, attention_heads=4)
    scheduler = DiffusionScheduler(timesteps=10, schedule="cosine")
    sample = DiffusionPipeline(model, scheduler).sample(
        1, 8, device="cpu", inference_steps=3, eta=0.0,
    )
    assert sample.shape == (1, 3, 8, 8)
    assert torch.isfinite(sample).all()


def test_classifier_free_guidance_sampling() -> None:
    model = SmallUNet(image_channels=3, base_channels=8, condition_size=16)
    pipeline = DiffusionPipeline(model, DiffusionScheduler(timesteps=4))
    sample = pipeline.sample(
        2, 8, device="cpu", text_condition=torch.randn(2, 16),
        guidance_scale=3.0, inference_steps=2,
    )
    assert sample.shape == (2, 3, 8, 8)


def test_class_conditioning_training_and_guided_sampling() -> None:
    model = SmallUNet(
        image_channels=3, base_channels=8, condition_size=16, num_classes=4,
    )
    pipeline = DiffusionPipeline(model, DiffusionScheduler(timesteps=4))
    labels = torch.tensor([1, 3])
    loss = pipeline.training_loss(
        torch.randn(2, 3, 8, 8), class_labels=labels, condition_dropout=0.5,
    )
    loss.backward()
    assert model.class_embedding.weight.grad is not None
    sample = pipeline.sample(
        2, 8, device="cpu", class_labels=labels, guidance_scale=3.0,
        inference_steps=2,
    )
    assert sample.shape == (2, 3, 8, 8)


def test_class_conditioning_requires_configured_classes() -> None:
    model = SmallUNet(image_channels=3, base_channels=8, condition_size=16)
    with pytest.raises(ValueError, match="num_classes"):
        model(torch.randn(1, 3, 8, 8), torch.tensor([1]), class_labels=torch.tensor([0]))


def test_pipeline_rejects_invalid_conditioning_options() -> None:
    model = SmallUNet(image_channels=3, base_channels=8, condition_size=16, num_classes=3)
    pipeline = DiffusionPipeline(model, DiffusionScheduler(timesteps=4))
    images = torch.randn(1, 3, 8, 8)
    with pytest.raises(ValueError, match="condition_dropout"):
        pipeline.training_loss(images, condition_dropout=1.1)
    with pytest.raises(ValueError, match="outside"):
        pipeline.training_loss(images, class_labels=torch.tensor([3]))
    with pytest.raises(ValueError, match="guidance_scale"):
        pipeline.sample(1, 8, device="cpu", guidance_scale=-1)
    with pytest.raises(ValueError, match="only one"):
        pipeline.training_loss(
            images, torch.randn(1, 16), class_labels=torch.tensor([1])
        )


def test_vae_reconstructs_and_has_differentiable_kl_loss() -> None:
    vae = AutoencoderKL(base_channels=8, latent_channels=4, downsample_factor=4)
    images = torch.randn(2, 3, 16, 16).clamp(-1, 1)
    output = vae(images)
    assert output.reconstruction.shape == images.shape
    assert output.mean.shape == (2, 4, 4, 4)
    loss = vae.loss(output, images)
    loss.backward()
    assert any(parameter.grad is not None for parameter in vae.parameters())


def test_text_encoder_and_cross_attention_condition_latent_diffusion() -> None:
    vae = AutoencoderKL(base_channels=8, latent_channels=4, downsample_factor=4)
    text_encoder = DiffusionTextEncoder(
        vocab_size=32, hidden_size=16, layers=1, heads=4, max_length=8,
    )
    model = SmallUNet(
        image_channels=4, base_channels=8, condition_size=16,
        use_cross_attention=True, attention_heads=4,
    )
    pipeline = LatentDiffusionPipeline(
        vae, model, DiffusionScheduler(timesteps=4), text_encoder,
    )
    images = torch.randn(2, 3, 32, 32).clamp(-1, 1)
    token_ids = torch.randint(1, 32, (2, 6))
    mask = torch.ones_like(token_ids, dtype=torch.bool)
    loss = pipeline.training_loss(images, token_ids=token_ids, attention_mask=mask)
    loss.backward()
    assert model.middle_cross_attention.attention.in_proj_weight.grad is not None
    generated = pipeline.sample(
        2, 32, device="cpu", token_ids=token_ids, attention_mask=mask,
        inference_steps=2,
    )
    assert generated.shape == (2, 3, 32, 32)
