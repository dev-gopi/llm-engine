"""Top-k/top-p token sampling."""

from __future__ import annotations

import math

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
        min_p: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        if not isinstance(logits, Tensor) or not logits.is_floating_point():
            raise TypeError("logits must be a floating-point tensor")
        if logits.ndim != 2 or logits.size(-1) == 0:
            raise ValueError("logits must have shape [batch, vocabulary]")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 0:
            raise ValueError("top_k must be a non-negative integer")
        if not 0 < top_p <= 1:
            raise ValueError("top_p must satisfy 0 < top_p <= 1")
        if not math.isfinite(min_p) or not 0 <= min_p <= 1:
            raise ValueError("min_p must satisfy 0 <= min_p <= 1")
        # Validate both greedy and stochastic paths. Negative infinity is a
        # legitimate blocked token, but each row must retain a usable token.
        if torch.isnan(logits).any() or torch.isposinf(logits).any():
            raise ValueError("logits cannot contain NaN or positive infinity")
        if not torch.isfinite(logits).any(dim=-1).all():
            raise ValueError("each logits row must contain an unblocked finite token")
        filtered = apply_temperature(logits.float(), temperature)
        if temperature == 0:
            return logits.argmax(dim=-1)
        if min_p:
            # p(token) >= min_p * p(best), expressed in logit space.
            threshold = filtered.amax(dim=-1, keepdim=True) + math.log(min_p)
            filtered = filtered.masked_fill(filtered < threshold, float("-inf"))
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
