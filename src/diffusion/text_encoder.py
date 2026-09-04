"""Tokenizer-agnostic Transformer text encoder for diffusion conditioning."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn


class DiffusionTextEncoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int = 256, layers: int = 4,
                 heads: int = 8, max_length: int = 128, dropout: float = 0.0,
                 padding_idx: int = 0) -> None:
        super().__init__()
        if hidden_size % heads:
            raise ValueError("hidden_size must be divisible by heads")
        if min(vocab_size, hidden_size, layers, heads, max_length) <= 0:
            raise ValueError("text encoder sizes must be positive")
        self.max_length = max_length
        self.hidden_size = hidden_size
        self.padding_idx = padding_idx
        self.token_embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=padding_idx)
        self.position_embedding = nn.Embedding(max_length, hidden_size)
        block = nn.TransformerEncoderLayer(
            hidden_size, heads, hidden_size * 4, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, token_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        if token_ids.ndim != 2 or token_ids.shape[1] > self.max_length:
            raise ValueError("token_ids must have shape [batch, sequence] within max_length")
        if token_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("token_ids must use an integer dtype")
        if attention_mask is None:
            attention_mask = token_ids.ne(self.padding_idx)
        if attention_mask.shape != token_ids.shape:
            raise ValueError("attention_mask must match token_ids")
        positions = torch.arange(token_ids.shape[1], device=token_ids.device)
        hidden = self.token_embedding(token_ids) + self.position_embedding(positions)[None]
        hidden = self.encoder(hidden, src_key_padding_mask=~attention_mask.bool())
        return self.norm(hidden)

    @classmethod
    def from_config(cls, config: Mapping[str, Any], *, vocab_size: int) -> "DiffusionTextEncoder":
        return cls(
            vocab_size, hidden_size=int(config.get("text_hidden_size", config.get("condition_size", 256))),
            layers=int(config.get("text_layers", 4)), heads=int(config.get("text_heads", 8)),
            max_length=int(config.get("text_max_length", 128)),
            dropout=float(config.get("text_dropout", 0.0)),
            padding_idx=int(config.get("text_padding_idx", 0)),
        )
