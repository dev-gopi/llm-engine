"""Training loss and sampling loop for the local scratch diffusion model."""

from __future__ import annotations

import math
import torch
from torch import Tensor
import torch.nn.functional as F

from .scheduler import DiffusionScheduler
from .unet import SmallUNet


class DiffusionPipeline:
    def __init__(self, model: SmallUNet, scheduler: DiffusionScheduler) -> None:
        self.model = model
        self.scheduler = scheduler

    def training_loss(self, images: Tensor, text_condition: Tensor | None = None,
                      *, condition_dropout: float = 0.0,
                      generator: torch.Generator | None = None,
                      class_labels: Tensor | None = None,
                      text_context: Tensor | None = None,
                      text_context_mask: Tensor | None = None) -> Tensor:
        if not 0 <= condition_dropout <= 1:
            raise ValueError("condition_dropout must be between zero and one")
        if sum(value is not None for value in (text_condition, class_labels, text_context)) > 1:
            raise ValueError("provide only one conditioning input")
        self._validate_class_labels(class_labels, images.shape[0])
        timesteps = torch.randint(
            0, self.scheduler.timesteps, (images.shape[0],), device=images.device,
            generator=generator,
        )
        noise = torch.randn(images.shape, device=images.device, dtype=images.dtype,
                            generator=generator)
        noisy, target_noise = self.scheduler.add_noise(images, timesteps, noise)
        if text_condition is not None and condition_dropout:
            keep = torch.rand(images.shape[0], device=images.device,
                              generator=generator) >= condition_dropout
            text_condition = text_condition * keep[:, None]
        if class_labels is not None and condition_dropout:
            if self.model.null_class_id is None:
                raise ValueError("class conditioning requires a class-conditional model")
            keep = torch.rand(images.shape[0], device=images.device,
                              generator=generator) >= condition_dropout
            class_labels = torch.where(
                keep, class_labels,
                torch.full_like(class_labels, self.model.null_class_id),
            )
        if text_context is not None and condition_dropout:
            keep = torch.rand(images.shape[0], device=images.device,
                              generator=generator) >= condition_dropout
            text_context = text_context * keep[:, None, None]
        predicted_noise = self.model(
            noisy, timesteps, text_condition, class_labels,
            text_context=text_context, text_context_mask=text_context_mask,
        )
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
        inference_steps: int | None = None,
        guidance_scale: float = 1.0,
        eta: float = 0.0,
        class_labels: Tensor | None = None,
        text_context: Tensor | None = None,
        text_context_mask: Tensor | None = None,
    ) -> Tensor:
        if batch_size <= 0 or image_size <= 0 or image_size % 4:
            raise ValueError("batch_size must be positive and image_size divisible by four")
        if not math.isfinite(guidance_scale) or guidance_scale < 0:
            raise ValueError("guidance_scale must be finite and non-negative")
        if sum(value is not None for value in (text_condition, class_labels, text_context)) > 1:
            raise ValueError("provide only one conditioning input")
        self._validate_class_labels(class_labels, batch_size)
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
            use_ddim = inference_steps is not None
            step_count = inference_steps or self.scheduler.timesteps
            if not 1 <= step_count <= self.scheduler.timesteps:
                raise ValueError("inference_steps must be between one and scheduler timesteps")
            schedule = torch.linspace(
                self.scheduler.timesteps - 1, 0, step_count, dtype=torch.long
            ).unique_consecutive().tolist()
            for index, timestep in enumerate(schedule):
                steps = torch.full((batch_size,), timestep, dtype=torch.long, device=device)
                predicted_noise = self.model(
                    sample, steps, text_condition, class_labels,
                    text_context=text_context, text_context_mask=text_context_mask,
                )
                if text_condition is not None and guidance_scale != 1.0:
                    unconditional = self.model(sample, steps, torch.zeros_like(text_condition))
                    predicted_noise = unconditional + guidance_scale * (predicted_noise - unconditional)
                if class_labels is not None and guidance_scale != 1.0:
                    if self.model.null_class_id is None:
                        raise ValueError("class conditioning requires a class-conditional model")
                    null_labels = torch.full_like(class_labels, self.model.null_class_id)
                    unconditional = self.model(sample, steps, class_labels=null_labels)
                    predicted_noise = unconditional + guidance_scale * (predicted_noise - unconditional)
                if text_context is not None and guidance_scale != 1.0:
                    unconditional = self.model(
                        sample, steps, text_context=torch.zeros_like(text_context),
                        text_context_mask=text_context_mask,
                    )
                    predicted_noise = unconditional + guidance_scale * (predicted_noise - unconditional)
                if use_ddim:
                    previous = schedule[index + 1] if index + 1 < len(schedule) else -1
                    sample = self.scheduler.ddim_step(
                        predicted_noise, timestep, previous, sample, eta=eta, generator=generator
                    )
                else:
                    sample = self.scheduler.step(
                        predicted_noise, timestep, sample, generator=generator
                    )
        finally:
            self.model.train(was_training)
        return sample.clamp(-1.0, 1.0)

    def _validate_class_labels(self, labels: Tensor | None, batch_size: int) -> None:
        if labels is None:
            return
        if self.model.num_classes is None:
            raise ValueError("class_labels require num_classes in the model configuration")
        if labels.shape != (batch_size,) or labels.dtype not in (torch.int32, torch.int64):
            raise ValueError("class_labels must be an integer tensor with shape [batch]")
        if labels.numel() and (int(labels.min()) < 0 or int(labels.max()) >= self.model.num_classes):
            raise ValueError("class_labels contain an ID outside the configured classes")
