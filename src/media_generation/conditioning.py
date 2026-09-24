"""Tokenizer-backed text conditioning shared by audio and video diffusion."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from diffusion.text_encoder import DiffusionTextEncoder
from tokenizer.encoder import Tokenizer


@dataclass(frozen=True)
class TokenBatch:
    ids: Tensor
    mask: Tensor


@dataclass(frozen=True)
class TextConditioning:
    """Pooled and token-level text features for FiLM + cross-attention."""

    pooled: Tensor
    context: Tensor
    mask: Tensor


class TextConditioner(nn.Module):
    """Trainable text encoder with pooled and token-level conditioning outputs."""

    def __init__(self, encoder: DiffusionTextEncoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.output_size = encoder.hidden_size

    def encode_features(self, token_ids: Tensor, attention_mask: Tensor) -> TextConditioning:
        context = self.encoder(token_ids, attention_mask)
        weights = attention_mask.to(context.dtype).unsqueeze(-1)
        pooled = (context * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        return TextConditioning(pooled=pooled, context=context, mask=attention_mask)

    def forward(self, token_ids: Tensor, attention_mask: Tensor) -> Tensor:
        # Preserve the original pooled-only API for backwards compatibility.
        return self.encode_features(token_ids, attention_mask).pooled

    @torch.inference_mode()
    def encode_prompts(
        self,
        prompts: Sequence[str],
        tokenizer: Tokenizer,
        *,
        device: torch.device | str,
        return_features: bool = False,
    ) -> Tensor | TextConditioning:
        batch = tokenize_prompts(prompts, tokenizer, max_length=self.encoder.max_length, device=device)
        features = self.encode_features(batch.ids, batch.mask)
        return features if return_features else features.pooled


def tokenize_prompts(
    prompts: Sequence[str],
    tokenizer: Tokenizer,
    *,
    max_length: int,
    device: torch.device | str | None = None,
) -> TokenBatch:
    if not prompts or any(not isinstance(prompt, str) for prompt in prompts):
        raise ValueError("prompts must be a non-empty sequence of strings")
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    sequences = [tokenizer.encode(prompt, add_bos=True, add_eos=True, allowed_special="all")[:max_length] for prompt in prompts]
    width = max(1, max(len(sequence) for sequence in sequences))
    padding_id = tokenizer.special_tokens.get("<|pad|>", 0)
    ids = torch.full((len(sequences), width), padding_id, dtype=torch.long, device=device)
    mask = torch.zeros((len(sequences), width), dtype=torch.bool, device=device)
    for row, sequence in enumerate(sequences):
        if sequence:
            ids[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=device)
            mask[row, : len(sequence)] = True
    return TokenBatch(ids, mask)


def build_text_conditioner(config: Mapping[str, Any], tokenizer: Tokenizer) -> TextConditioner:
    return TextConditioner(DiffusionTextEncoder.from_config(config, vocab_size=tokenizer.vocab_size))
