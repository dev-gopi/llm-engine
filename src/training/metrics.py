"""Training and evaluation metrics."""

from __future__ import annotations

import math

from torch import Tensor


def perplexity(loss: float | Tensor) -> float | Tensor:
    """Convert mean token cross-entropy (natural log) to perplexity."""

    if isinstance(loss, Tensor):
        return loss.exp()
    return math.exp(loss)
