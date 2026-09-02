import pytest
import torch

from diffusion.pipeline import DiffusionPipeline
from diffusion.scheduler import DiffusionScheduler
from diffusion.unet import SmallUNet


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
