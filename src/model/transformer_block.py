"""A single Transformer block."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from .attention import MultiHeadAttention
from .feed_forward import FeedForward


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 8) -> None:
        super().__init__()
        self.attn = MultiHeadAttention(dim, heads)
        self.ffn = FeedForward(dim)
        self.n1 = nn.LayerNorm(dim)
        self.n2 = nn.LayerNorm(dim)

    def forward(self, hidden_states: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        hidden_states = self.n1(
            hidden_states + self.attn(hidden_states, attention_mask=attention_mask)
        )
        hidden_states = self.n2(hidden_states + self.ffn(hidden_states))
        return hidden_states
