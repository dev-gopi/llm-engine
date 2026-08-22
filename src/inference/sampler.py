"""Top-k/top-p token sampling."""

from __future__ import annotations

import torch
from torch import Tensor

from .temperature import apply_temperature


class TopKSampler:
    def __call__(
        self,
        logits: Tensor,
        *,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        if logits.ndim != 2:
            raise ValueError("logits must have shape [batch, vocabulary]")
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        if not 0 < top_p <= 1:
            raise ValueError("top_p must satisfy 0 < top_p <= 1")
        if temperature == 0:
            return logits.argmax(dim=-1)

        filtered = apply_temperature(logits.float(), temperature)
        if top_k:
            k = min(top_k, filtered.size(-1))
            threshold = filtered.topk(k, dim=-1).values[:, -1, None]
            filtered = filtered.masked_fill(filtered < threshold, float("-inf"))
        if top_p < 1.0:
            sorted_logits, sorted_indices = filtered.sort(dim=-1, descending=True)
            cumulative = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
            remove = cumulative > top_p
            remove[:, 1:] = remove[:, :-1].clone()
            remove[:, 0] = False
            sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
            filtered = torch.full_like(filtered, float("-inf")).scatter(
                1, sorted_indices, sorted_logits
            )
        return torch.multinomial(filtered.softmax(dim=-1), 1, generator=generator).squeeze(1)
