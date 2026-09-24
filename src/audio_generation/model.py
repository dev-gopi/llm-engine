"""Latent audio autoencoder and text-conditioned 1-D diffusion denoiser."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from media_generation.conditioning import TextConditioner
from media_generation.nn import FiLM, timestep_embedding, valid_group_count


class AudioAutoencoder1D(nn.Module):
    """Convolutional audio autoencoder with configurable latent compression."""

    def __init__(self, *, channels: int = 96, latent_channels: int = 64, downsample_stages: int = 6) -> None:
        super().__init__()
        if min(channels, latent_channels, downsample_stages) <= 0:
            raise ValueError("audio autoencoder dimensions must be positive")
        encoder: list[nn.Module] = [nn.Conv1d(1, channels, 7, padding=3), nn.SiLU()]
        current = channels
        for stage in range(downsample_stages):
            next_channels = min(channels * (2 ** min(stage + 1, 3)), channels * 8)
            encoder += [
                nn.Conv1d(current, next_channels, 4, stride=2, padding=1),
                nn.GroupNorm(valid_group_count(next_channels), next_channels),
                nn.SiLU(),
            ]
            current = next_channels
        encoder.append(nn.Conv1d(current, latent_channels, 3, padding=1))
        self.encoder = nn.Sequential(*encoder)

        decoder: list[nn.Module] = [nn.Conv1d(latent_channels, current, 3, padding=1), nn.SiLU()]
        reversed_channels: list[int] = []
        value = channels
        for stage in range(downsample_stages):
            value = min(channels * (2 ** min(stage + 1, 3)), channels * 8)
            reversed_channels.append(value)
        for output_channels in reversed([channels] + reversed_channels[:-1]):
            decoder += [
                nn.ConvTranspose1d(current, output_channels, 4, stride=2, padding=1),
                nn.GroupNorm(valid_group_count(output_channels), output_channels),
                nn.SiLU(),
            ]
            current = output_channels
        decoder.append(nn.Conv1d(current, 1, 7, padding=3))
        self.decoder = nn.Sequential(*decoder)
        self.downsample_factor = 2 ** downsample_stages
        self.latent_channels = latent_channels

    def encode(self, waveform: Tensor) -> Tensor:
        if waveform.ndim == 2:
            waveform = waveform[:, None]
        if waveform.ndim != 3 or waveform.shape[1] != 1:
            raise ValueError("waveform must have shape [batch, samples] or [batch, 1, samples]")
        return self.encoder(waveform)

    def decode(self, latents: Tensor, *, target_samples: int | None = None) -> Tensor:
        waveform = torch.tanh(self.decoder(latents))
        if target_samples is not None:
            waveform = waveform[..., :target_samples]
        return waveform

    def forward(self, waveform: Tensor) -> tuple[Tensor, Tensor]:
        latents = self.encode(waveform)
        return self.decode(latents, target_samples=waveform.shape[-1]), latents


class AudioCrossAttention(nn.Module):
    """Cross-attention from latent audio positions to token-level text context."""

    def __init__(self, channels: int, context_size: int, heads: int = 8) -> None:
        super().__init__()
        if channels % heads:
            raise ValueError("audio cross-attention channels must be divisible by heads")
        self.norm = nn.LayerNorm(channels)
        self.context_norm = nn.LayerNorm(context_size)
        self.attention = nn.MultiheadAttention(
            channels, heads, kdim=context_size, vdim=context_size, batch_first=True
        )

    def forward(self, x: Tensor, context: Tensor, context_mask: Tensor | None = None) -> Tensor:
        sequence = x.transpose(1, 2)
        query = self.norm(sequence)
        text = self.context_norm(context)
        key_padding_mask = None if context_mask is None else ~context_mask.bool()
        attended, _ = self.attention(
            query, text, text, key_padding_mask=key_padding_mask, need_weights=False
        )
        return (sequence + attended).transpose(1, 2)


class AudioResidualBlock(nn.Module):
    def __init__(self, channels: int, embedding_size: int, *, dilation: int, dropout: float) -> None:
        super().__init__()
        groups = valid_group_count(channels)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.film = FiLM(embedding_size, channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)

    def forward(self, x: Tensor, embedding: Tensor) -> Tensor:
        hidden = self.conv1(F.silu(self.norm1(x)))
        hidden = self.film(self.norm2(hidden), embedding)
        hidden = self.conv2(self.dropout(F.silu(hidden)))
        return x + hidden


class AudioDenoiser1D(nn.Module):
    """WaveNet-like latent denoiser with timestep and pooled-text FiLM conditioning."""

    def __init__(self, latent_channels: int, *, model_channels: int = 256, blocks: int = 16,
                 condition_size: int = 512, dropout: float = 0.05,
                 cross_attention_every: int = 0, cross_attention_heads: int = 8) -> None:
        super().__init__()
        if min(latent_channels, model_channels, blocks, condition_size) <= 0:
            raise ValueError("audio denoiser dimensions must be positive")
        self.condition_size = condition_size
        self.embedding_size = model_channels * 4
        self.input = nn.Conv1d(latent_channels, model_channels, 3, padding=1)
        self.time_mlp = nn.Sequential(
            nn.Linear(model_channels, self.embedding_size), nn.SiLU(),
            nn.Linear(self.embedding_size, self.embedding_size),
        )
        self.condition_mlp = nn.Sequential(
            nn.Linear(condition_size, self.embedding_size), nn.SiLU(),
            nn.Linear(self.embedding_size, self.embedding_size),
        )
        self.blocks = nn.ModuleList([
            AudioResidualBlock(model_channels, self.embedding_size, dilation=2 ** (index % 8), dropout=dropout)
            for index in range(blocks)
        ])
        self.cross_attention = nn.ModuleDict({
            str(index): AudioCrossAttention(model_channels, condition_size, cross_attention_heads)
            for index in range(blocks)
            if cross_attention_every > 0 and (index + 1) % cross_attention_every == 0
        })
        self.output = nn.Sequential(
            nn.GroupNorm(valid_group_count(model_channels), model_channels), nn.SiLU(),
            nn.Conv1d(model_channels, latent_channels, 3, padding=1),
        )

    def forward(
        self, noisy_latents: Tensor, timesteps: Tensor, condition: Tensor | None = None,
        *, context: Tensor | None = None, context_mask: Tensor | None = None,
    ) -> Tensor:
        if noisy_latents.ndim != 3:
            raise ValueError("audio latents must have shape [batch, channels, time]")
        time = timestep_embedding(timesteps, self.input.out_channels).to(noisy_latents.dtype)
        embedding = self.time_mlp(time)
        if condition is not None:
            if condition.shape != (noisy_latents.shape[0], self.condition_size):
                raise ValueError("audio condition shape does not match [batch, condition_size]")
            embedding = embedding + self.condition_mlp(condition.to(embedding.dtype))
        hidden = self.input(noisy_latents)
        for index, block in enumerate(self.blocks):
            hidden = block(hidden, embedding)
            key = str(index)
            if key in self.cross_attention and context is not None:
                hidden = self.cross_attention[key](hidden, context, context_mask)
        return self.output(hidden)


class AudioDiffusionModel(nn.Module):
    """Checkpointable text encoder + audio autoencoder + latent denoiser."""

    def __init__(self, autoencoder: AudioAutoencoder1D, denoiser: AudioDenoiser1D,
                 text_conditioner: TextConditioner) -> None:
        super().__init__()
        if denoiser.condition_size != text_conditioner.output_size:
            raise ValueError("denoiser condition size must equal text encoder hidden size")
        self.autoencoder = autoencoder
        self.denoiser = denoiser
        self.text_conditioner = text_conditioner

    @classmethod
    def from_config(cls, config: Mapping[str, Any], *, text_conditioner: TextConditioner) -> "AudioDiffusionModel":
        autoencoder = AudioAutoencoder1D(
            channels=int(config.get("autoencoder_channels", 96)),
            latent_channels=int(config.get("latent_channels", 64)),
            downsample_stages=int(config.get("downsample_stages", 6)),
        )
        denoiser = AudioDenoiser1D(
            autoencoder.latent_channels,
            model_channels=int(config.get("model_channels", 256)),
            blocks=int(config.get("blocks", 16)),
            condition_size=text_conditioner.output_size,
            dropout=float(config.get("dropout", 0.05)),
            cross_attention_every=int(config.get("cross_attention_every", 0)),
            cross_attention_heads=int(config.get("cross_attention_heads", 8)),
        )
        return cls(autoencoder, denoiser, text_conditioner)
