"""Latent video autoencoder with factorized temporal attention diffusion denoiser."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from media_generation.conditioning import TextConditioner
from media_generation.nn import FiLM, timestep_embedding, valid_group_count


class VideoAutoencoder3D(nn.Module):
    """Compact spatiotemporal autoencoder for RGB clips."""

    def __init__(self, *, channels: int = 64, latent_channels: int = 16,
                 spatial_stages: int = 3, temporal_downsample: bool = True) -> None:
        super().__init__()
        if min(channels, latent_channels, spatial_stages) <= 0:
            raise ValueError("video autoencoder dimensions must be positive")
        encoder: list[nn.Module] = [nn.Conv3d(3, channels, 3, padding=1), nn.SiLU()]
        current = channels
        strides: list[tuple[int, int, int]] = []
        for stage in range(spatial_stages):
            stride = (2 if temporal_downsample and stage == 0 else 1, 2, 2)
            strides.append(stride)
            next_channels = min(channels * (2 ** (stage + 1)), channels * 8)
            encoder += [
                nn.Conv3d(current, next_channels, (4 if stride[0] == 2 else 3, 4, 4), stride=stride, padding=1),
                nn.GroupNorm(valid_group_count(next_channels), next_channels), nn.SiLU(),
            ]
            current = next_channels
        encoder.append(nn.Conv3d(current, latent_channels, 3, padding=1))
        self.encoder = nn.Sequential(*encoder)
        decoder: list[nn.Module] = [nn.Conv3d(latent_channels, current, 3, padding=1), nn.SiLU()]
        stage_channels = [min(channels * (2 ** (stage + 1)), channels * 8) for stage in range(spatial_stages)]
        outputs = list(reversed([channels] + stage_channels[:-1]))
        for stride, output_channels in zip(reversed(strides), outputs):
            decoder += [
                nn.ConvTranspose3d(current, output_channels, (4 if stride[0] == 2 else 3, 4, 4), stride=stride, padding=1),
                nn.GroupNorm(valid_group_count(output_channels), output_channels), nn.SiLU(),
            ]
            current = output_channels
        decoder.append(nn.Conv3d(current, 3, 3, padding=1))
        self.decoder = nn.Sequential(*decoder)
        self.spatial_factor = 2 ** spatial_stages
        self.temporal_factor = 2 if temporal_downsample else 1
        self.latent_channels = latent_channels

    def encode(self, video: Tensor) -> Tensor:
        if video.ndim != 5 or video.shape[1] != 3:
            raise ValueError("video must have shape [batch, 3, frames, height, width]")
        return self.encoder(video)

    def decode(self, latents: Tensor, *, target_shape: tuple[int, int, int] | None = None) -> Tensor:
        video = torch.tanh(self.decoder(latents))
        if target_shape is not None:
            video = video[..., : target_shape[0], : target_shape[1], : target_shape[2]]
        return video

    def forward(self, video: Tensor) -> tuple[Tensor, Tensor]:
        latents = self.encode(video)
        return self.decode(latents, target_shape=tuple(video.shape[-3:])), latents


class TemporalAttention(nn.Module):
    """Self-attention over time independently at every spatial location."""

    def __init__(self, channels: int, heads: int = 8) -> None:
        super().__init__()
        if channels % heads:
            raise ValueError("temporal attention channels must be divisible by heads")
        self.norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(channels, heads, batch_first=True)

    def forward(self, x: Tensor) -> Tensor:
        batch, channels, frames, height, width = x.shape
        sequence = x.permute(0, 3, 4, 2, 1).reshape(batch * height * width, frames, channels)
        normalized = self.norm(sequence)
        attended, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        sequence = sequence + attended
        return sequence.reshape(batch, height, width, frames, channels).permute(0, 4, 3, 1, 2)


class VideoCrossAttention(nn.Module):
    """Memory-bounded cross-attention from video latent tokens to text context."""

    def __init__(self, channels: int, context_size: int, heads: int = 8, chunk_size: int = 2048) -> None:
        super().__init__()
        if channels % heads:
            raise ValueError("video cross-attention channels must be divisible by heads")
        if chunk_size <= 0:
            raise ValueError("video cross-attention chunk size must be positive")
        self.norm = nn.LayerNorm(channels)
        self.context_norm = nn.LayerNorm(context_size)
        self.chunk_size = chunk_size
        self.attention = nn.MultiheadAttention(
            channels, heads, kdim=context_size, vdim=context_size, batch_first=True
        )

    def forward(self, x: Tensor, context: Tensor, context_mask: Tensor | None = None) -> Tensor:
        batch, channels, frames, height, width = x.shape
        sequence = x.permute(0, 2, 3, 4, 1).reshape(batch, frames * height * width, channels)
        text = self.context_norm(context)
        key_padding_mask = None if context_mask is None else ~context_mask.bool()
        outputs = []
        for start in range(0, sequence.shape[1], self.chunk_size):
            chunk = sequence[:, start:start + self.chunk_size]
            attended, _ = self.attention(
                self.norm(chunk), text, text, key_padding_mask=key_padding_mask, need_weights=False
            )
            outputs.append(chunk + attended)
        sequence = torch.cat(outputs, dim=1)
        return sequence.reshape(batch, frames, height, width, channels).permute(0, 4, 1, 2, 3)


class VideoResidualBlock(nn.Module):
    def __init__(self, channels: int, embedding_size: int, dropout: float) -> None:
        super().__init__()
        groups = valid_group_count(channels)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.film = FiLM(embedding_size, channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1)

    def forward(self, x: Tensor, embedding: Tensor) -> Tensor:
        hidden = self.conv1(F.silu(self.norm1(x)))
        hidden = self.film(self.norm2(hidden), embedding)
        hidden = self.conv2(self.dropout(F.silu(hidden)))
        return x + hidden


class VideoDenoiser3D(nn.Module):
    """3-D latent denoiser with temporal attention and text/timestep conditioning."""

    def __init__(self, latent_channels: int, *, model_channels: int = 192, blocks: int = 12,
                 condition_size: int = 512, temporal_heads: int = 8, attention_every: int = 3,
                 dropout: float = 0.05, cross_attention_every: int = 0,
                 cross_attention_heads: int = 8, cross_attention_chunk_size: int = 2048) -> None:
        super().__init__()
        self.condition_size = condition_size
        self.embedding_size = model_channels * 4
        self.input = nn.Conv3d(latent_channels, model_channels, 3, padding=1)
        self.time_mlp = nn.Sequential(nn.Linear(model_channels, self.embedding_size), nn.SiLU(), nn.Linear(self.embedding_size, self.embedding_size))
        self.condition_mlp = nn.Sequential(nn.Linear(condition_size, self.embedding_size), nn.SiLU(), nn.Linear(self.embedding_size, self.embedding_size))
        self.blocks = nn.ModuleList([VideoResidualBlock(model_channels, self.embedding_size, dropout) for _ in range(blocks)])
        self.attention = nn.ModuleDict({str(index): TemporalAttention(model_channels, temporal_heads) for index in range(blocks) if attention_every > 0 and (index + 1) % attention_every == 0})
        self.cross_attention = nn.ModuleDict({
            str(index): VideoCrossAttention(
                model_channels, condition_size, cross_attention_heads, cross_attention_chunk_size
            )
            for index in range(blocks)
            if cross_attention_every > 0 and (index + 1) % cross_attention_every == 0
        })
        self.output = nn.Sequential(nn.GroupNorm(valid_group_count(model_channels), model_channels), nn.SiLU(), nn.Conv3d(model_channels, latent_channels, 3, padding=1))

    def forward(
        self, noisy_latents: Tensor, timesteps: Tensor, condition: Tensor | None = None,
        *, context: Tensor | None = None, context_mask: Tensor | None = None,
    ) -> Tensor:
        if noisy_latents.ndim != 5:
            raise ValueError("video latents must have shape [batch, channels, frames, height, width]")
        time = timestep_embedding(timesteps, self.input.out_channels).to(noisy_latents.dtype)
        embedding = self.time_mlp(time)
        if condition is not None:
            if condition.shape != (noisy_latents.shape[0], self.condition_size):
                raise ValueError("video condition shape does not match [batch, condition_size]")
            embedding = embedding + self.condition_mlp(condition.to(embedding.dtype))
        hidden = self.input(noisy_latents)
        for index, block in enumerate(self.blocks):
            hidden = block(hidden, embedding)
            key = str(index)
            if key in self.attention:
                hidden = self.attention[key](hidden)
            if key in self.cross_attention and context is not None:
                hidden = self.cross_attention[key](hidden, context, context_mask)
        return self.output(hidden)


class VideoDiffusionModel(nn.Module):
    def __init__(self, autoencoder: VideoAutoencoder3D, denoiser: VideoDenoiser3D,
                 text_conditioner: TextConditioner) -> None:
        super().__init__()
        if denoiser.condition_size != text_conditioner.output_size:
            raise ValueError("denoiser condition size must equal text encoder hidden size")
        self.autoencoder = autoencoder
        self.denoiser = denoiser
        self.text_conditioner = text_conditioner

    @classmethod
    def from_config(cls, config: Mapping[str, Any], *, text_conditioner: TextConditioner) -> "VideoDiffusionModel":
        autoencoder = VideoAutoencoder3D(
            channels=int(config.get("autoencoder_channels", 64)),
            latent_channels=int(config.get("latent_channels", 16)),
            spatial_stages=int(config.get("spatial_stages", 3)),
            temporal_downsample=bool(config.get("temporal_downsample", True)),
        )
        denoiser = VideoDenoiser3D(
            autoencoder.latent_channels,
            model_channels=int(config.get("model_channels", 192)),
            blocks=int(config.get("blocks", 12)),
            condition_size=text_conditioner.output_size,
            temporal_heads=int(config.get("temporal_heads", 8)),
            attention_every=int(config.get("attention_every", 3)),
            dropout=float(config.get("dropout", 0.05)),
            cross_attention_every=int(config.get("cross_attention_every", 0)),
            cross_attention_heads=int(config.get("cross_attention_heads", 8)),
            cross_attention_chunk_size=int(config.get("cross_attention_chunk_size", 2048)),
        )
        return cls(autoencoder, denoiser, text_conditioner)
