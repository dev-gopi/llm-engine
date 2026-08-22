"""Learned absolute positional embeddings for GPT-style models."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from utils.logger import get_logger

logger = get_logger(__name__)


class PositionalEmbedding(nn.Module):
    """Trainable positions with explicit IDs, decoding offsets, and padding."""

    def __init__(
        self,
        max_pos: int,
        dim: int,
        *,
        initializer_range: float = 0.02,
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
        active = positions if mask is None else positions[mask]
        if active.numel() == 0:
            return
        minimum, maximum = int(active.min()), int(active.max())
        if minimum < 0 or maximum >= self.max_positions:
            raise IndexError(
                f"position IDs must be in [0, {self.max_positions - 1}], "
                f"but received range [{minimum}, {maximum}]"
            )

    def resize(self, new_max_positions: int) -> None:
        """Resize the table while preserving all overlapping weights."""
        if not isinstance(new_max_positions, int) or isinstance(new_max_positions, bool) or new_max_positions <= 0:
            raise ValueError("new_max_positions must be a positive integer")
        if new_max_positions == self.max_positions:
            return
        old = self.emb
        replacement = nn.Embedding(
            new_max_positions, self.embedding_dim, device=old.weight.device, dtype=old.weight.dtype
        )
        nn.init.normal_(replacement.weight, mean=0.0, std=self.initializer_range)
        with torch.no_grad():
            replacement.weight[: min(self.max_positions, new_max_positions)].copy_(
                old.weight[: min(self.max_positions, new_max_positions)]
            )
        self.emb = replacement
        previous = self.max_positions
        self.max_positions = new_max_positions
        logger.info("Resized positional embeddings from %d to %d", previous, new_max_positions)

    @classmethod
    def from_config(cls, config: Mapping[str, Any], **kwargs: Any) -> "PositionalEmbedding":
        return cls(
            max_pos=int(config["max_position"]),
            dim=int(config["hidden_size"]),
            initializer_range=float(
                config.get("position_initializer_range", config.get("initializer_range", 0.02))
            ),
            **kwargs,
        )
