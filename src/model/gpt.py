"""Complete GPT-style language model."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from .embedding import TokenEmbedding
from .positional import PositionalEmbedding
from .transformer_block import TransformerBlock


class MiniGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int = 128,
        layers: int = 4,
        heads: int = 4,
        max_pos: int = 512,
    ) -> None:
        super().__init__()
        self.tok = TokenEmbedding(vocab_size, dim)
        self.pos = PositionalEmbedding(max_pos, dim)
        self.blocks = nn.ModuleList(
            TransformerBlock(dim, heads) for _ in range(layers)
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, token_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        hidden_states = self.tok(token_ids) + self.pos(token_ids)
        for block in self.blocks:
            hidden_states = block(hidden_states, attention_mask=attention_mask)
        return self.head(self.norm(hidden_states))
