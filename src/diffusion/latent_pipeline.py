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
                      condition_dropout: float = 0.1) -> Tensor:
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
            condition_dropout=condition_dropout,
        )

    @torch.inference_mode()
    def sample(self, batch_size: int, image_size: int, *, device: torch.device | str,
               token_ids: Tensor | None = None, attention_mask: Tensor | None = None,
               guidance_scale: float = 5.0, inference_steps: int = 50,
               generator: torch.Generator | None = None) -> Tensor:
        if image_size % self.vae.downsample_factor:
            raise ValueError("image_size must be divisible by the VAE downsample factor")
        context = None
        if token_ids is not None:
            if self.text_encoder is None:
                raise ValueError("token_ids require a text encoder")
            context = self.text_encoder(token_ids, attention_mask)
        latent_size = image_size // self.vae.downsample_factor
        latents = self.diffusion.sample(
            batch_size, latent_size, device=device, text_context=context,
            text_context_mask=attention_mask, guidance_scale=guidance_scale,
            inference_steps=inference_steps, generator=generator,
        )
        return self.vae.decode(latents / self.latent_scale).clamp(-1, 1)
