"""Positional embedding mechanisms including learned, rotary (RoPE), and sinusoidal embeddings."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from utils.logger import get_logger

logger = get_logger(__name__)


def rotate_half(x: Tensor) -> Tensor:
    """Rotates half the hidden dims of the input tensor for RoPE."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: Tensor,
    k: Tensor,
    cos: Tensor,
    sin: Tensor,
    position_ids: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Apply rotary position embedding (RoPE) to query and key tensors.

    q, k shape: [batch, heads, seq, head_dim]
    cos, sin shape: [1, 1, seq, head_dim] or [batch, 1, seq, head_dim]
    """
    if position_ids is not None:
        # Index cos and sin using position_ids
        # position_ids shape: [batch, seq] or [1, seq]
        cos = cos.squeeze(0).squeeze(0)[position_ids].unsqueeze(1)  # [batch, 1, seq, head_dim]
        sin = sin.squeeze(0).squeeze(0)[position_ids].unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) with optional position scaling / interpolation."""

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        scaling_factor: float = 1.0,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(dim, int) or dim <= 0 or dim % 2 != 0:
            raise ValueError("dim must be a positive even integer for RoPE")
        if max_position_embeddings <= 0:
            raise ValueError("max_position_embeddings must be positive")
        if base <= 0:
            raise ValueError("base must be positive")
        if scaling_factor <= 0:
            raise ValueError("scaling_factor must be positive")

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = float(base)
        self.scaling_factor = float(scaling_factor)

        # Inverse frequencies
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._build_cos_sin_cache(max_position_embeddings, device=device, dtype=dtype)

    def _build_cos_sin_cache(
        self, seq_len: int, device: torch.device | str | None = None, dtype: torch.dtype | None = None
    ) -> None:
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        if self.scaling_factor != 1.0:
            t = t / self.scaling_factor

        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype=dtype, device=device), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype=dtype, device=device), persistent=False)

    def forward(self, x: Tensor, seq_len: int | None = None) -> tuple[Tensor, Tensor]:
        """Return (cos, sin) tensors cached up to max(seq_len, x.shape[2])."""
        target_seq_len = seq_len if seq_len is not None else x.shape[2]
        if target_seq_len > self.max_seq_len_cached:
            self._build_cos_sin_cache(target_seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:, :, :target_seq_len, :].to(dtype=x.dtype, device=x.device),
            self.sin_cached[:, :, :target_seq_len, :].to(dtype=x.dtype, device=x.device),
        )


class SinusoidalPositionalEmbedding(nn.Module):
    """Fixed sinusoidal positional embedding."""

    def __init__(
        self,
        max_pos: int,
        dim: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.max_positions = max_pos
        self.embedding_dim = dim

        position = torch.arange(max_pos, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))

        pe = torch.zeros(max_pos, dim, dtype=dtype, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("weight", pe, persistent=True)

    def forward(self, inputs: Tensor, position_offset: int = 0) -> Tensor:
        batch_size, sequence_length = inputs.shape[:2]
        end_pos = position_offset + sequence_length
        if end_pos > self.max_positions:
            raise IndexError(f"End position {end_pos} exceeds max_positions {self.max_positions}")
        embeddings = self.weight[position_offset:end_pos].unsqueeze(0).expand(batch_size, -1, -1)
        return embeddings.to(dtype=inputs.dtype, device=inputs.device)


class PositionalEmbedding(nn.Module):
    """Trainable positions with explicit IDs, decoding offsets, and padding."""

    def __init__(
        self,
        max_pos: int,
        dim: int,
        *,
        initializer_range: float = 0.02,
        interpolate_positions: bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(max_pos, int) or isinstance(max_pos, bool) or max_pos <= 0:
            raise ValueError("max_pos must be a positive integer")
        if not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0:
            raise ValueError("dim must be a positive integer")
        if initializer_range <= 0:
            raise ValueError("initializer_range must be greater than zero")
        self.max_positions = max_pos
        self.embedding_dim = dim
        self.initializer_range = float(initializer_range)
        self.interpolate_positions = bool(interpolate_positions)
        self.emb = nn.Embedding(max_pos, dim, device=device, dtype=dtype)
        self.reset_parameters()

    @property
    def weight(self) -> nn.Parameter:
        return self.emb.weight

    def reset_parameters(self) -> None:
        nn.init.normal_(self.emb.weight, mean=0.0, std=self.initializer_range)

    def forward(
        self,
        inputs: Tensor,
        *,
        position_ids: Tensor | None = None,
        position_offset: int = 0,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """Return embeddings shaped ``[batch, sequence, embedding_dim]``."""
        if inputs.ndim not in (2, 3):
            raise ValueError("inputs must have shape [batch, sequence] or [batch, sequence, dim]")
        batch_size, sequence_length = inputs.shape[:2]
        if not isinstance(position_offset, int) or isinstance(position_offset, bool):
            raise TypeError("position_offset must be an integer")
        if position_offset < 0:
            raise ValueError("position_offset must be non-negative")

        mask = self._validate_mask(attention_mask, batch_size, sequence_length, inputs.device)
        positions = self._make_position_ids(
            batch_size, sequence_length, inputs.device, position_ids, position_offset, mask
        )

        if self.interpolate_positions and not torch.compiler.is_compiling():
            max_p = positions.max().item()
            if max_p >= self.max_positions:
                positions = (positions.float() * (self.max_positions - 1) / float(max_p)).long()

        self._validate_bounds(positions, mask)
        embeddings = self.emb(positions)
        if mask is not None:
            embeddings = embeddings * mask.unsqueeze(-1).to(embeddings.dtype)
        return embeddings

    @staticmethod
    def _validate_mask(
        attention_mask: Tensor | None,
        batch_size: int,
        sequence_length: int,
        device: torch.device,
    ) -> Tensor | None:
        if attention_mask is None:
            return None
        if attention_mask.shape != (batch_size, sequence_length):
            raise ValueError("attention_mask must have shape [batch, sequence]")
        if attention_mask.device != device:
            raise ValueError("attention_mask and inputs must be on the same device")
        if attention_mask.dtype != torch.bool and not torch.all(
            (attention_mask == 0) | (attention_mask == 1)
        ):
            raise ValueError("attention_mask values must be binary")
        return attention_mask.bool()

    @staticmethod
    def _make_position_ids(
        batch_size: int,
        sequence_length: int,
        device: torch.device,
        position_ids: Tensor | None,
        position_offset: int,
        attention_mask: Tensor | None,
    ) -> Tensor:
        if position_ids is not None:
            if position_ids.dtype not in (torch.int32, torch.int64):
                raise TypeError("position_ids must use an integer dtype")
            if position_ids.ndim == 1 and position_ids.shape[0] == sequence_length:
                return position_ids.to(device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
            if position_ids.shape == (batch_size, sequence_length):
                return position_ids.to(device=device, dtype=torch.long)
            raise ValueError("position_ids must have shape [sequence] or [batch, sequence]")
        if attention_mask is not None:
            return (attention_mask.long().cumsum(dim=1) - 1 + position_offset).clamp_min(0)
        positions = torch.arange(
            position_offset, position_offset + sequence_length, device=device, dtype=torch.long
        )
        return positions.unsqueeze(0).expand(batch_size, -1)

    def _validate_bounds(self, positions: Tensor, mask: Tensor | None) -> None:
        if torch.compiler.is_compiling():
            return
        active = positions if mask is None else positions[mask]
        if active.numel() == 0:
            return
        minimum, maximum = int(active.min()), int(active.max())
        if minimum < 0 or maximum >= self.max_positions:
            raise IndexError(
                f"position IDs must be in [0, {self.max_positions - 1}], "
                f"but received range [{minimum}, {maximum}]"
            )

    def resize(self, new_max_positions: int, *, interpolate: bool = False) -> None:
        """Resize the table while preserving or interpolating weights."""
        if not isinstance(new_max_positions, int) or isinstance(new_max_positions, bool) or new_max_positions <= 0:
            raise ValueError("new_max_positions must be a positive integer")
        if new_max_positions == self.max_positions:
            return
        old = self.emb
        replacement = nn.Embedding(
            new_max_positions, self.embedding_dim, device=old.weight.device, dtype=old.weight.dtype
        )
        with torch.no_grad():
            if interpolate and self.max_positions > 1:
                # Interpolate old weights across new position range
                weight_reshaped = old.weight.t().unsqueeze(0)  # [1, dim, old_pos]
                interpolated = torch.nn.functional.interpolate(
                    weight_reshaped, size=new_max_positions, mode="linear", align_corners=True
                )
                replacement.weight.copy_(interpolated.squeeze(0).t())
            else:
                nn.init.normal_(replacement.weight, mean=0.0, std=self.initializer_range)
                replacement.weight[: min(self.max_positions, new_max_positions)].copy_(
                    old.weight[: min(self.max_positions, new_max_positions)]
                )
        self.emb = replacement
        previous = self.max_positions
        self.max_positions = new_max_positions
        logger.info("Resized positional embeddings from %d to %d (interpolate=%s)", previous, new_max_positions, interpolate)

    @classmethod
    def from_config(cls, config: Mapping[str, Any], **kwargs: Any) -> "PositionalEmbedding":
        return cls(
            max_pos=int(config["max_position"]),
            dim=int(config["hidden_size"]),
            initializer_range=float(
                config.get("position_initializer_range", config.get("initializer_range", 0.02))
            ),
            interpolate_positions=bool(config.get("interpolate_positions", False)),
            **kwargs,
        )
