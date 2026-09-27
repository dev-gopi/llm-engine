"""Latent-space diffusion training and decoding pipeline."""

from __future__ import annotations

import torch
from torch import Tensor

from .pipeline import DiffusionPipeline
from .scheduler import DiffusionScheduler
from .text_encoder import DiffusionTextEncoder
from .unet import SmallUNet
from .vae import AutoencoderKL


class LatentDiffusionPipeline:
    def __init__(self, vae: AutoencoderKL, model: SmallUNet,
                 scheduler: DiffusionScheduler,
                 text_encoder: DiffusionTextEncoder | None = None,
                 latent_scale: float = 0.18215) -> None:
        if model.image_channels != vae.latent_channels:
            raise ValueError("U-Net image_channels must match VAE latent_channels")
        if latent_scale <= 0:
            raise ValueError("latent_scale must be positive")
        self.vae = vae
        self.model = model
        self.scheduler = scheduler
        self.text_encoder = text_encoder
        self.latent_scale = latent_scale
        self.diffusion = DiffusionPipeline(model, scheduler)

    def training_loss(self, images: Tensor, *, token_ids: Tensor | None = None,
                      attention_mask: Tensor | None = None,
                      condition_dropout: float = 0.1,
                      min_snr_gamma: float | None = None) -> Tensor:
        with torch.no_grad():
            latents, _, _ = self.vae.encode(images)
            latents = latents * self.latent_scale
        context = None
        if token_ids is not None:
            if self.text_encoder is None:
                raise ValueError("token_ids require a text encoder")
            context = self.text_encoder(token_ids, attention_mask)
        return self.diffusion.training_loss(
            latents, text_context=context, text_context_mask=attention_mask,
            condition_dropout=condition_dropout, min_snr_gamma=min_snr_gamma,
        )

    @torch.inference_mode()
    def sample(self, batch_size: int, image_size: int, *, device: torch.device | str,
               token_ids: Tensor | None = None, attention_mask: Tensor | None = None,
               negative_token_ids: Tensor | None = None,
               negative_attention_mask: Tensor | None = None,
               guidance_scale: float = 5.0, inference_steps: int = 50,
               generator: torch.Generator | None = None) -> Tensor:
        if image_size % self.vae.downsample_factor:
            raise ValueError("image_size must be divisible by the VAE downsample factor")
        modules = [self.vae]
        if self.text_encoder is not None:
            modules.append(self.text_encoder)
        modes = [module.training for module in modules]
        for module in modules:
            module.eval()
        try:
            context = None
            if token_ids is not None:
                if self.text_encoder is None:
                    raise ValueError("token_ids require a text encoder")
                context = self.text_encoder(token_ids, attention_mask)
            negative_context = None
            if negative_token_ids is not None:
                if token_ids is None:
                    raise ValueError("negative_token_ids require token_ids")
                if self.text_encoder is None:
                    raise ValueError("negative_token_ids require a text encoder")
                negative_context = self.text_encoder(negative_token_ids, negative_attention_mask)
            elif negative_attention_mask is not None:
                raise ValueError("negative_attention_mask requires negative_token_ids")
            latent_size = image_size // self.vae.downsample_factor
            latents = self.diffusion.sample(
                batch_size, latent_size, device=device, text_context=context,
                text_context_mask=attention_mask, guidance_scale=guidance_scale,
                negative_text_context=negative_context,
                negative_text_context_mask=negative_attention_mask,
                inference_steps=inference_steps, generator=generator,
            )
            return self.vae.decode(latents / self.latent_scale).clamp(-1, 1)
        finally:
            for module, was_training in zip(modules, modes, strict=True):
                module.train(was_training)
    @torch.inference_mode()
    def edit(self, images: Tensor, *, device: torch.device | str, strength: float = 0.5,
             token_ids: Tensor | None = None, attention_mask: Tensor | None = None,
             negative_token_ids: Tensor | None = None, negative_attention_mask: Tensor | None = None,
             guidance_scale: float = 5.0, inference_steps: int = 50,
             generator: torch.Generator | None = None, eta: float = 0.0) -> Tensor:
        """Image-to-image latent diffusion using the same trained checkpoint.

        ``strength=0`` is an exact reconstruction path (subject to VAE loss), while
        ``strength=1`` applies the full configured denoising schedule.
        """
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape [batch, 3, height, width]")
        if images.shape[-1] != images.shape[-2]:
            raise ValueError("native latent editing currently requires square images")
        if images.shape[-1] % self.vae.downsample_factor:
            raise ValueError("image size must be divisible by the VAE downsample factor")
        if not 0.0 <= strength <= 1.0:
            raise ValueError("strength must be between zero and one")
        if not 1 <= inference_steps <= self.scheduler.timesteps:
            raise ValueError("inference_steps must be between one and scheduler timesteps")
        batch = images.shape[0]
        if token_ids is not None and token_ids.shape[0] != batch:
            raise ValueError("token_ids batch must match images")
        if negative_token_ids is not None and negative_token_ids.shape != token_ids.shape:
            raise ValueError("negative_token_ids must match token_ids")
        modules = [self.vae]
        if self.text_encoder is not None:
            modules.append(self.text_encoder)
        modes = [m.training for m in modules]
        for m in modules:
            m.eval()
        try:
            source = images.to(device)
            latents, _, _ = self.vae.encode(source)
            latents = latents * self.latent_scale
            if strength == 0.0:
                return self.vae.decode(latents / self.latent_scale).clamp(-1, 1)
            context = None
            negative_context = None
            if token_ids is not None:
                if self.text_encoder is None:
                    raise ValueError("token_ids require a text encoder")
                context = self.text_encoder(token_ids, attention_mask)
                if negative_token_ids is not None:
                    negative_context = self.text_encoder(negative_token_ids, negative_attention_mask)
            schedule = torch.linspace(self.scheduler.timesteps - 1, 0, inference_steps, dtype=torch.long).unique_consecutive().tolist()
            start = min(len(schedule) - 1, max(0, round((1.0 - strength) * (len(schedule) - 1))))
            schedule = schedule[start:]
            first_t = int(schedule[0])
            self.scheduler.to(device)
            noise = torch.randn(latents.shape, device=device, dtype=latents.dtype, generator=generator)
            steps = torch.full((batch,), first_t, dtype=torch.long, device=device)
            sample, _ = self.scheduler.add_noise(latents, steps, noise)
            null_context = negative_context if negative_context is not None else (torch.zeros_like(context) if context is not None else None)
            was_training = self.model.training
            self.model.eval()
            try:
                for index, timestep in enumerate(schedule):
                    ts = torch.full((batch,), int(timestep), dtype=torch.long, device=device)
                    predicted = self.model(sample, ts, text_context=context, text_context_mask=attention_mask)
                    if context is not None and guidance_scale != 1.0:
                        unconditional = self.model(
                            sample, ts, text_context=null_context,
                            text_context_mask=(negative_attention_mask if negative_context is not None else attention_mask),
                        )
                        predicted = unconditional + guidance_scale * (predicted - unconditional)
                    previous = int(schedule[index + 1]) if index + 1 < len(schedule) else -1
                    sample = self.scheduler.ddim_step(predicted, int(timestep), previous, sample, eta=eta, generator=generator)
            finally:
                self.model.train(was_training)
            return self.vae.decode(sample / self.latent_scale).clamp(-1, 1)
        finally:
            for module, was_training in zip(modules, modes, strict=True):
                module.train(was_training)

