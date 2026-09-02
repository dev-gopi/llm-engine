"""Training loss and sampling loop for the local scratch diffusion model."""

from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F

from .scheduler import DiffusionScheduler
from .unet import SmallUNet


class DiffusionPipeline:
    def __init__(self, model: SmallUNet, scheduler: DiffusionScheduler) -> None:
        self.model = model
        self.scheduler = scheduler

    def training_loss(self, images: Tensor, text_condition: Tensor | None = None) -> Tensor:
        timesteps = torch.randint(
            0, self.scheduler.timesteps, (images.shape[0],), device=images.device
        )
        noisy, target_noise = self.scheduler.add_noise(images, timesteps)
        predicted_noise = self.model(noisy, timesteps, text_condition)
        return F.mse_loss(predicted_noise.float(), target_noise.float())

    @torch.inference_mode()
    def sample(
        self,
        batch_size: int,
        image_size: int,
        *,
        device: torch.device | str,
        text_condition: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        if batch_size <= 0 or image_size <= 0 or image_size % 4:
            raise ValueError("batch_size must be positive and image_size divisible by four")
        sample = torch.randn(
            batch_size,
            self.model.image_channels,
            image_size,
            image_size,
            device=device,
            generator=generator,
        )
        self.scheduler.to(device)
        was_training = self.model.training
        self.model.eval()
        try:
            for timestep in reversed(range(self.scheduler.timesteps)):
                steps = torch.full((batch_size,), timestep, dtype=torch.long, device=device)
                predicted_noise = self.model(sample, steps, text_condition)
                sample = self.scheduler.step(
                    predicted_noise, timestep, sample, generator=generator
                )
        finally:
            self.model.train(was_training)
        return sample.clamp(-1.0, 1.0)
