"""Training and DDIM sampling for text-conditioned latent audio diffusion."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from diffusion.scheduler import DiffusionScheduler
from media_generation.production import diffusion_loss, make_noise
from media_generation.progress import CancellationToken, ProgressCallback
from .model import AudioDiffusionModel


class AudioGenerationPipeline:
    def __init__(self, model: AudioDiffusionModel, scheduler: DiffusionScheduler) -> None:
        self.model = model
        self.scheduler = scheduler

    def training_loss(
        self,
        waveform: Tensor,
        condition: Tensor,
        *,
        context: Tensor | None = None,
        context_mask: Tensor | None = None,
        condition_dropout: float = 0.1,
        reconstruction_weight: float = 0.1,
        min_snr_gamma: float = 0.0,
        noise_offset: float = 0.0,
        input_perturbation: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if not 0 <= condition_dropout <= 1:
            raise ValueError("condition_dropout must be between zero and one")
        reconstruction, latents = self.model.autoencoder(waveform)
        target_waveform = waveform[:, None] if waveform.ndim == 2 else waveform
        reconstruction_loss = F.l1_loss(reconstruction, target_waveform)
        if condition_dropout:
            keep = torch.rand(condition.shape[0], device=condition.device, generator=generator) >= condition_dropout
            condition = condition * keep[:, None].to(condition.dtype)
            if context is not None:
                context = context * keep[:, None, None].to(context.dtype)
        timesteps = torch.randint(
            0, self.scheduler.timesteps, (latents.shape[0],), device=latents.device, generator=generator
        )
        target_noise, noisy_noise = make_noise(
            latents,
            generator=generator,
            noise_offset=noise_offset,
            input_perturbation=input_perturbation,
        )
        noisy, _ = self.scheduler.add_noise(latents, timesteps, noise=noisy_noise)
        prediction = self.model.denoiser(
            noisy, timesteps, condition, context=context, context_mask=context_mask
        )
        d_loss = diffusion_loss(
            prediction,
            target_noise,
            alpha_bars=self.scheduler.alpha_bars,
            timesteps=timesteps,
            min_snr_gamma=min_snr_gamma,
        )
        total = d_loss + reconstruction_weight * reconstruction_loss
        return total, {"diffusion": d_loss.detach(), "reconstruction": reconstruction_loss.detach()}

    @torch.inference_mode()
    def sample(
        self,
        condition: Tensor,
        *,
        samples: int,
        sample_rate: int,
        device: torch.device | str,
        context: Tensor | None = None,
        context_mask: Tensor | None = None,
        negative_context: Tensor | None = None,
        negative_context_mask: Tensor | None = None,
        guidance_scale: float = 4.5,
        inference_steps: int = 50,
        eta: float = 0.0,
        seed: int | None = None,
        negative_condition: Tensor | None = None,
        init_waveform: Tensor | None = None,
        strength: float = 1.0,
        preserve_mask: Tensor | None = None,
        progress_callback: ProgressCallback | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> Tensor:
        if samples <= 0 or sample_rate <= 0:
            raise ValueError("samples and sample_rate must be positive")
        if inference_steps < 1 or inference_steps > self.scheduler.timesteps:
            raise ValueError("inference_steps must be within the diffusion schedule")
        if not 0.0 <= strength <= 1.0:
            raise ValueError("strength must be between zero and one")
        batch = condition.shape[0]
        factor = self.model.autoencoder.downsample_factor
        latent_length = math.ceil(samples / factor)
        generator = torch.Generator(device=device)
        if seed is not None:
            generator.manual_seed(seed)
        steps = torch.linspace(
            self.scheduler.timesteps - 1, 0, inference_steps, device=device
        ).long().unique_consecutive()
        encoded_source = None
        latent_preserve_mask = None
        if init_waveform is None:
            latent = torch.randn(
                (batch, self.model.autoencoder.latent_channels, latent_length),
                device=device,
                generator=generator,
            )
            start_index = 0
        else:
            initial = init_waveform.to(device)
            if initial.ndim == 1:
                initial = initial[None]
            if initial.shape[0] == 1 and batch > 1:
                initial = initial.expand(batch, -1)
            if initial.shape[0] != batch:
                raise ValueError("init_waveform batch size must match prompt batch size")
            if initial.shape[-1] < samples:
                initial = F.pad(initial, (0, samples - initial.shape[-1]))
            initial = initial[..., :samples]
            encoded = self.model.autoencoder.encode(initial)
            encoded_source = encoded
            if preserve_mask is not None:
                mask = preserve_mask.to(device=device, dtype=encoded.dtype)
                if mask.ndim == 1:
                    mask = mask[None, None]
                elif mask.ndim == 2:
                    mask = mask[:, None]
                if mask.ndim != 3:
                    raise ValueError("preserve_mask must have shape [samples], [batch,samples], or [batch,1,samples]")
                if mask.shape[0] == 1 and batch > 1:
                    mask = mask.expand(batch, -1, -1)
                if mask.shape[0] != batch:
                    raise ValueError("preserve_mask batch size must match prompt batch size")
                latent_preserve_mask = F.interpolate(mask.clamp(0, 1), size=encoded.shape[-1], mode="linear", align_corners=False)
            if strength == 0:
                return self.model.autoencoder.decode(encoded, target_samples=samples)[:, 0].clamp(-1, 1)
            start_index = min(len(steps) - 1, max(0, round((1.0 - strength) * (len(steps) - 1))))
            timestep = int(steps[start_index].item())
            noise = torch.randn(encoded.shape, device=device, dtype=encoded.dtype, generator=generator)
            t = torch.full((batch,), timestep, device=device, dtype=torch.long)
            latent, _ = self.scheduler.add_noise(encoded, t, noise=noise)
        null_condition = torch.zeros_like(condition) if negative_condition is None else negative_condition
        null_context = torch.zeros_like(context) if context is not None and negative_context is None else negative_context
        null_context_mask = context_mask if negative_context_mask is None else negative_context_mask
        was_training = self.model.training
        self.model.eval()
        try:
            total_steps = len(steps) - start_index
            for index in range(start_index, len(steps)):
                if cancellation_token is not None and cancellation_token.cancelled:
                    raise RuntimeError("generation cancelled")
                step_tensor = steps[index]
                timestep = int(step_tensor.item())
                previous = int(steps[index + 1].item()) if index + 1 < len(steps) else -1
                t = torch.full((batch,), timestep, device=device, dtype=torch.long)
                conditional = self.model.denoiser(
                    latent, t, condition, context=context, context_mask=context_mask
                )
                if guidance_scale != 1.0:
                    unconditional = self.model.denoiser(
                        latent,
                        t,
                        null_condition,
                        context=null_context,
                        context_mask=null_context_mask,
                    )
                    prediction = unconditional + guidance_scale * (conditional - unconditional)
                else:
                    prediction = conditional
                latent = self.scheduler.ddim_step(
                    prediction, timestep, previous, latent, eta=eta, generator=generator
                )
                if encoded_source is not None and latent_preserve_mask is not None:
                    if previous >= 0:
                        preserve_t = torch.full((batch,), previous, device=device, dtype=torch.long)
                        source_noise = torch.randn(encoded_source.shape, device=device, dtype=encoded_source.dtype, generator=generator)
                        preserved, _ = self.scheduler.add_noise(encoded_source, preserve_t, noise=source_noise)
                    else:
                        preserved = encoded_source
                    latent = latent * (1 - latent_preserve_mask) + preserved * latent_preserve_mask
                if progress_callback is not None:
                    progress_callback(index - start_index + 1, total_steps)
            return self.model.autoencoder.decode(latent, target_samples=samples)[:, 0].clamp(-1, 1)
        finally:
            self.model.train(was_training)
