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
