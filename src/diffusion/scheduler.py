"""DDPM forward process and reverse-process coefficients."""

from __future__ import annotations

import math
import torch
from torch import Tensor


class DiffusionScheduler:
    def __init__(
        self,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        *,
        device: torch.device | str | None = None,
        schedule: str = "linear",
    ) -> None:
        if timesteps < 2:
            raise ValueError("timesteps must be at least two")
        if not 0.0 < beta_start < beta_end < 1.0:
            raise ValueError("betas must satisfy 0 < beta_start < beta_end < 1")
        if schedule not in {"linear", "cosine"}:
            raise ValueError("schedule must be linear or cosine")
        self.timesteps = timesteps
        self.schedule = schedule
        if schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        else:
            steps = torch.linspace(0, timesteps, timesteps + 1, device=device)
            alpha_bar = torch.cos(((steps / timesteps + 0.008) / 1.008) * math.pi / 2).square()
            alpha_bar = alpha_bar / alpha_bar[0]
            self.betas = (1 - alpha_bar[1:] / alpha_bar[:-1]).clamp(1e-5, 0.999)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def to(self, device: torch.device | str) -> "DiffusionScheduler":
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        return self

    @staticmethod
    def _extract(values: Tensor, timesteps: Tensor, target: Tensor) -> Tensor:
        selected = values.to(timesteps.device)[timesteps]
        return selected.view(-1, *((1,) * (target.ndim - 1))).to(target.dtype)

    def add_noise(
        self, clean_images: Tensor, timesteps: Tensor, noise: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        self._validate_timesteps(timesteps, clean_images.shape[0])
        noise = torch.randn_like(clean_images) if noise is None else noise
        if noise.shape != clean_images.shape:
            raise ValueError("noise shape must match clean_images")
        alpha_bar = self._extract(self.alpha_bars, timesteps, clean_images)
        noisy = alpha_bar.sqrt() * clean_images + (1.0 - alpha_bar).sqrt() * noise
        return noisy, noise

    def step(
        self,
        predicted_noise: Tensor,
        timestep: int,
        sample: Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        if predicted_noise.shape != sample.shape:
            raise ValueError("predicted_noise and sample shapes must match")
        if not 0 <= timestep < self.timesteps:
            raise ValueError("timestep is outside the diffusion schedule")
        beta = self.betas[timestep].to(device=sample.device, dtype=sample.dtype)
        alpha = self.alphas[timestep].to(device=sample.device, dtype=sample.dtype)
        alpha_bar = self.alpha_bars[timestep].to(device=sample.device, dtype=sample.dtype)
        mean = (sample - beta * predicted_noise / (1.0 - alpha_bar).sqrt()) / alpha.sqrt()
        if timestep == 0:
            return mean
        previous_alpha_bar = self.alpha_bars[timestep - 1].to(
            device=sample.device, dtype=sample.dtype
        )
        posterior_variance = beta * (1.0 - previous_alpha_bar) / (1.0 - alpha_bar)
        noise = torch.randn(
            sample.shape,
            device=sample.device,
            dtype=sample.dtype,
            generator=generator,
        )
        return mean + posterior_variance.clamp_min(0).sqrt() * noise

    def ddim_step(
        self, predicted_noise: Tensor, timestep: int, previous_timestep: int,
        sample: Tensor, *, eta: float = 0.0, generator: torch.Generator | None = None,
    ) -> Tensor:
        """One DDIM update, allowing fewer inference steps than training."""
        if not 0.0 <= eta <= 1.0:
            raise ValueError("eta must satisfy 0 <= eta <= 1")
        alpha_bar = self.alpha_bars[timestep].to(sample)
        previous_alpha_bar = (
            self.alpha_bars[previous_timestep].to(sample)
            if previous_timestep >= 0 else torch.ones((), device=sample.device, dtype=sample.dtype)
        )
        clean = (sample - (1 - alpha_bar).sqrt() * predicted_noise) / alpha_bar.sqrt()
        variance = ((1 - previous_alpha_bar) / (1 - alpha_bar) *
                    (1 - alpha_bar / previous_alpha_bar)).clamp_min(0)
        sigma = eta * variance.sqrt()
        direction = (1 - previous_alpha_bar - sigma.square()).clamp_min(0).sqrt() * predicted_noise
        if previous_timestep < 0 or eta == 0:
            noise = 0.0
        else:
            noise = torch.randn(sample.shape, device=sample.device, dtype=sample.dtype,
                                generator=generator)
        return previous_alpha_bar.sqrt() * clean + direction + sigma * noise

    def _validate_timesteps(self, timesteps: Tensor, batch_size: int) -> None:
        if timesteps.ndim != 1 or timesteps.shape[0] != batch_size:
            raise ValueError("timesteps must have shape [batch]")
        if timesteps.dtype not in (torch.int32, torch.int64):
            raise TypeError("timesteps must use an integer dtype")
        if timesteps.numel() and (int(timesteps.min()) < 0 or int(timesteps.max()) >= self.timesteps):
            raise ValueError("timesteps contain an index outside the diffusion schedule")
