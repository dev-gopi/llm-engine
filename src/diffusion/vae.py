"""Compact convolutional variational autoencoder for latent diffusion."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class VAEOutput:
    reconstruction: Tensor
    mean: Tensor
    log_variance: Tensor


class AutoencoderKL(nn.Module):
    """Encode images into spatial Gaussian latents and reconstruct them."""

    def __init__(self, image_channels: int = 3, latent_channels: int = 4,
                 base_channels: int = 64, downsample_factor: int = 4) -> None:
        super().__init__()
        if downsample_factor not in {2, 4, 8}:
            raise ValueError("downsample_factor must be 2, 4, or 8")
        if min(image_channels, latent_channels, base_channels) <= 0:
            raise ValueError("channel sizes must be positive")
        self.image_channels = image_channels
        self.latent_channels = latent_channels
        self.downsample_factor = downsample_factor
        levels = downsample_factor.bit_length() - 1
        encoder: list[nn.Module] = [nn.Conv2d(image_channels, base_channels, 3, padding=1), nn.SiLU()]
        channels = base_channels
        for _ in range(levels):
            encoder.extend((nn.Conv2d(channels, channels * 2, 4, stride=2, padding=1), nn.SiLU()))
            channels *= 2
        encoder.append(nn.Conv2d(channels, latent_channels * 2, 3, padding=1))
        self.encoder = nn.Sequential(*encoder)
        decoder: list[nn.Module] = [nn.Conv2d(latent_channels, channels, 3, padding=1), nn.SiLU()]
        for _ in range(levels):
            decoder.extend((nn.ConvTranspose2d(channels, channels // 2, 4, stride=2, padding=1), nn.SiLU()))
            channels //= 2
        decoder.extend((nn.Conv2d(channels, image_channels, 3, padding=1), nn.Tanh()))
        self.decoder = nn.Sequential(*decoder)

    def encode(self, images: Tensor, *, sample: bool = True,
               generator: torch.Generator | None = None) -> tuple[Tensor, Tensor, Tensor]:
        if images.ndim != 4 or images.shape[1] != self.image_channels:
            raise ValueError("images have an incompatible shape")
        if images.shape[-2] % self.downsample_factor or images.shape[-1] % self.downsample_factor:
            raise ValueError("image dimensions must be divisible by downsample_factor")
        mean, log_variance = self.encoder(images).chunk(2, dim=1)
        log_variance = log_variance.clamp(-30, 20)
        if sample:
            noise = torch.randn(mean.shape, device=mean.device, dtype=mean.dtype, generator=generator)
            latent = mean + (0.5 * log_variance).exp() * noise
        else:
            latent = mean
        return latent, mean, log_variance

    def decode(self, latents: Tensor) -> Tensor:
        if latents.ndim != 4 or latents.shape[1] != self.latent_channels:
            raise ValueError("latents have an incompatible shape")
        return self.decoder(latents)

    def forward(self, images: Tensor, *, generator: torch.Generator | None = None) -> VAEOutput:
        latent, mean, log_variance = self.encode(images, generator=generator)
        return VAEOutput(self.decode(latent), mean, log_variance)

    @staticmethod
    def loss(output: VAEOutput, target: Tensor, *, kl_weight: float = 1e-6) -> Tensor:
        if kl_weight < 0:
            raise ValueError("kl_weight must be non-negative")
        reconstruction = F.l1_loss(output.reconstruction.float(), target.float())
        kl = -0.5 * (1 + output.log_variance - output.mean.square() - output.log_variance.exp())
        return reconstruction + kl_weight * kl.float().mean()

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "AutoencoderKL":
        return cls(
            image_channels=int(config.get("image_channels", 3)),
            latent_channels=int(config.get("latent_channels", 4)),
            base_channels=int(config.get("vae_base_channels", 64)),
            downsample_factor=int(config.get("vae_downsample_factor", 4)),
        )
